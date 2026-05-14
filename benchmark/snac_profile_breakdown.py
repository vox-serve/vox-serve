"""Where does the SNAC decoder spend its time at bs=64, compile+graph+fp16?

Uses torch.profiler over the captured CUDA graph replay to get per-kernel breakdown.
"""

import torch

import vox_serve.tokenizer.snac as snac_mod
from vox_serve.tokenizer.snac import NoiseBlock


def make_codes(bs, vocab=4096):
    g = torch.Generator().manual_seed(0)
    return [
        torch.randint(0, vocab, (bs, 4), generator=g).cuda().int(),
        torch.randint(0, vocab, (bs, 8), generator=g).cuda().int(),
        torch.randint(0, vocab, (bs, 16), generator=g).cuda().int(),
    ]


def main():
    torch.set_grad_enabled(False)
    snac_mod.FUSE_REMOVE_WEIGHT_NORM = True
    snac_mod.FUSE_COMPOSE_QUANTIZER_TABLE = True

    model = snac_mod.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda().to(torch.float16)
    for m in model.modules():
        if isinstance(m, NoiseBlock):
            m.forward = lambda x: x

    model.decoder = torch.compile(model.decoder, dynamic=True)
    model.quantizer.from_codes = torch.compile(model.quantizer.from_codes, dynamic=True)

    bs = 64
    codes = make_codes(bs)
    # Trigger compile
    for _ in range(3):
        _ = model.decode(codes)
    torch.cuda.synchronize()

    # Profile direct calls (not cuda-graph replay; profiler can't see into graph kernels by name).
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(20):
            _ = model.decode(codes)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
