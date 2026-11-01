"""Python-only correctness test of the captured graph programs (no Rust).

Builds the engine exactly as the Rust runtime does, then drives the captured graphs from Python
(FlashInfer planning through the wrappers, torch graph replay) with teacher forcing and compares the
greedy token of every step against HF transformers' logits for the same Llama weights. Mismatches
are reported with HF's top-1/top-2 logit margin so bf16 near-ties can be told apart from bugs.

    python rust/scripts/py_smoke.py --model canopylabs/orpheus-3b-0.1-ft --steps 64
"""

import argparse
import json

import torch

from vox_serve.rt import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="canopylabs/orpheus-3b-0.1-ft")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--text", default="Hello there, this is a quick test of the rust runtime.")
    args = ap.parse_args()

    cfg = {"model": args.model, "max_batch_size": 4, "prefill_buckets": [256], "max_context": 1024}
    spec = json.loads(core.build_engine(json.dumps(cfg)))
    program, builder = core._ENGINE_KEEPALIVE[-1]
    inp, out = builder.inputs, builder.outputs
    meta = spec["meta"]
    ps = meta["page_size"]
    n_entries = meta["max_batch_size"]
    dummy = program.max_slots
    graphs = {(g.stage, tuple(g.key)): g for g in builder.graphs}

    from transformers import AutoTokenizer, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model)
    template = meta["prompt_template"]
    prompt = template["prefix_ids"] + tok(f"tara: {args.text}").input_ids + template["suffix_ids"]
    L = len(prompt)

    ref = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()
    gen = ref.generate(
        torch.tensor([prompt], device="cuda"),
        max_new_tokens=args.steps,
        do_sample=False,
        repetition_penalty=1.0,
        attention_mask=torch.ones(1, L, dtype=torch.long, device="cuda"),
    )
    seq = gen[0].tolist()
    with torch.no_grad():
        logits = ref(torch.tensor([seq], device="cuda")).logits[0].float()
    top2 = logits.topk(2, dim=-1)
    ref_arg = top2.indices[:, 0].tolist()
    margin = (top2.values[:, 0] - top2.values[:, 1]).tolist()
    del ref
    torch.cuda.empty_cache()

    # Each "request" owns a slot and a contiguous run of pages.
    pages_per = (len(seq) + ps - 1) // ps

    def pages_of(slot):
        return list(range(1 + slot * pages_per, 1 + (slot + 1) * pages_per))

    stats = {"n": 0, "mismatch": 0, "bad": []}

    def check(where, pos, got):
        stats["n"] += 1
        if got != ref_arg[pos]:
            stats["mismatch"] += 1
            stats["bad"].append((where, pos, got, ref_arg[pos], round(margin[pos], 3)))

    def run(stage, key, entries):
        """entries: list of (slot, start_pos, n_rows, first); tokens come from `seq` (teacher forcing)."""
        g = graphs[(stage, key)]
        a = g.attention
        bs = a.batch_size
        rows_cap = key[0]
        inp.storage.zero_()
        inp["row_slot"].fill_(dummy)
        inp["slot"].fill_(dummy)
        inp["top_p"].fill_(1.0)
        inp["rep_pen"].fill_(1.0)
        qo, kv, idx, last = [0], [0], [], []
        row = 0
        for e in range(bs):
            if e < len(entries):
                slot, start, n, first = entries[e]
                pg = pages_of(slot)
                for k in range(n):
                    p = start + k
                    inp["tok"][row + k, 0] = seq[p]
                    inp["pos"][row + k] = p
                    inp["lm_kv_page"][row + k] = pg[p // ps]
                    inp["lm_kv_off"][row + k] = p % ps
                    inp["row_slot"][row + k] = slot
                kvl = start + n
                np_ = (kvl + ps - 1) // ps
                idx += pg[:np_]
                last.append(kvl - (np_ - 1) * ps)
                inp["last_row"][e] = row + n - 1
                inp["slot"][e] = slot
                inp["active"][e] = 1
                inp["first"][e] = first
                row += n
            elif a.kind == "prefill":
                last.append(1)
            else:
                idx.append(0)
                last.append(1)
                inp["last_row"][e] = row
                row += 1
            qo.append(row)
            kv.append(len(idx))
        assert row <= rows_cap
        qo_t = torch.tensor(qo, dtype=torch.int32)
        kv_t = torch.tensor(kv, dtype=torch.int32)
        idx_t = torch.tensor(idx, dtype=torch.int32)
        last_t = torch.tensor(last, dtype=torch.int32)
        if a.kind == "prefill":
            a.wrapper.plan(
                qo_indptr=qo_t,
                paged_kv_indptr=kv_t,
                paged_kv_indices=idx_t,
                paged_kv_last_page_len=last_t,
                num_qo_heads=a.num_qo_heads,
                num_kv_heads=a.num_kv_heads,
                head_dim_qk=a.head_dim,
                page_size=ps,
                causal=True,
                q_data_type=a.dtype,
                kv_data_type=a.dtype,
            )
        else:
            a.wrapper.plan(
                indptr=kv_t,
                indices=idx_t,
                last_page_len=last_t,
                num_qo_heads=a.num_qo_heads,
                num_kv_heads=a.num_kv_heads,
                head_dim=a.head_dim,
                page_size=ps,
                q_data_type=a.dtype,
                kv_data_type=a.dtype,
            )
        info = a.wrapper._plan_info
        info = info.tolist() if torch.is_tensor(info) else list(info)
        exported = [x for x in spec["graphs"] if x["stage"] == stage and x["key"] == list(key)][0]["attention"]
        assert info == exported["plan_info"], (info, exported["plan_info"])
        g.graph.replay()
        torch.cuda.synchronize()
        res = out["out_tok"][: len(entries), 0].tolist()
        for (slot, start, n, _first), t in zip(entries, res, strict=True):
            check(f"{stage}{list(key)} slot{slot}", start + n - 1, t)

    # 1) single request: prefill then teacher-forced decode with bs=1
    run("lm_prefill", (256,), [(0, 0, L, 1)])
    for p in range(L, len(seq) - 1):
        run("lm_decode", (1,), [(0, p, 1, 0)])
    single = dict(stats)
    print("single:", single["mismatch"], "/", single["n"], single["bad"][:8])

    # 2) batched: three prefills of different lengths in one prefill graph + one piggybacked decode
    stats.update(n=0, mismatch=0, bad=[])
    run("lm_prefill", (256,), [(1, 0, L, 1), (2, 0, L + 10, 1), (3, 0, L + 20, 1)])
    run("lm_prefill", (256,), [(0, 0, L + 5, 1)])
    pos = {0: L + 5, 1: L, 2: L + 10, 3: L + 20}
    while max(pos.values()) < len(seq) - 1:
        live = [s for s in range(4) if pos[s] < len(seq) - 1]
        bs = 1
        while bs < len(live):
            bs *= 2
        run("lm_decode", (bs,), [(s, pos[s], 1, 0) for s in live])
        for s in live:
            pos[s] += 1
    print("batched:", stats["mismatch"], "/", stats["n"], stats["bad"][:8])

    # 3) detok runs and produces non-silent PCM
    gd = graphs[("detok", (1,))]
    inp["dt_tok"][0, :, 0] = torch.tensor(seq[L : L + 28], dtype=torch.int32)
    gd.graph.replay()
    torch.cuda.synchronize()
    pcm = out["audio"][0]
    print("detok pcm mean |x|:", pcm.float().abs().mean().item())


if __name__ == "__main__":
    main()
