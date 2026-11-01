"""``vox-serve``: one command for the Rust and the Python serving runtime.

``--runtime auto`` (the default) serves with the Rust runtime (the ``vox_serve._rt`` extension) and falls back
to the Python runtime (``vox_serve.launch``) when the Rust one cannot serve the command line: the extension was
not built (its build is optional at install time) or does not load, it was built against another FlashInfer
version, or an argument exists only in the Python runtime. Without the extension, arguments only the Rust
runtime has are ignored with a warning. ``--runtime rust`` and ``--runtime python`` pick one runtime without
fallback. Errors after start-up never switch runtimes.
"""

import argparse
import importlib.util
import signal
import sys
from typing import List, Optional

PROG = "vox-serve"

HELP = f"""usage: {PROG} [--runtime {{auto,rust,python}}] [arguments of the runtime]

  --runtime auto    the Rust runtime, or the Python runtime when the Rust one is not installed or cannot serve
                    the arguments (default)
  --runtime rust    the Rust runtime only
  --runtime python  the Python runtime only ({PROG} --runtime python --help lists its arguments)
"""


def _load_rust():
    """(the Rust extension module, None), or (None, why it cannot be used)."""
    if importlib.util.find_spec("vox_serve._rt") is None:
        return None, "the Rust runtime is not installed (its optional build was skipped or failed at install time)"
    try:
        import flashinfer
        import torch  # noqa: F401  (loads libcudart first; the extension shares torch's CUDA runtime)

        from vox_serve import _rt
    except ImportError as e:
        return None, f"the Rust runtime does not load ({e})"
    built, installed = _rt.FLASHINFER_VERSION, flashinfer.__version__
    if built and built != installed:
        return None, f"the Rust runtime was built against FlashInfer {built}, but {installed} is installed"
    return _rt, None


def _python_accepts(args: List[str]) -> bool:
    from .launch import build_parser

    parser = build_parser()
    parser.exit_on_error = False
    try:
        _, unknown = parser.parse_known_args(args)
    except argparse.ArgumentError:
        return False
    return not unknown


def _python_args(args: List[str]) -> List[str]:
    """``args`` without the flags (and their values) the Python runtime does not know, with a warning."""
    from .launch import build_parser

    known = {a for action in build_parser()._actions for a in action.option_strings}
    kept, dropped, i = [], [], 0
    while i < len(args):
        name = args[i].split("=", 1)[0]
        if name.startswith("--") and name not in known:
            dropped.append(args[i])
            if "=" not in args[i] and i + 1 < len(args) and not args[i + 1].startswith("--"):
                dropped.append(args[i + 1])
                i += 1
        else:
            kept.append(args[i])
        i += 1
    if dropped:
        print(f"{PROG}: the Python runtime ignores these Rust-runtime arguments: {' '.join(dropped)}", file=sys.stderr)
    return kept


def _run_rust(rt, args: List[str]) -> None:
    # Ctrl-C: the HTTP server shuts down gracefully, anything else (start-up, bench) stops at once.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.stdout.flush()  # Rust writes to the file descriptors directly
    try:
        rt.serve([PROG, *args])
    except RuntimeError as e:
        sys.exit(f"{PROG}: {e}")


def _run_python(args: List[str], reason: Optional[str] = None) -> None:
    from .launch import main as python_main

    if reason:
        print(f"{PROG}: using the Python runtime: {reason}", file=sys.stderr)
    python_main(args, prog=PROG)


def main(argv: Optional[List[str]] = None) -> None:
    runtime_parser = argparse.ArgumentParser(prog=PROG, add_help=False, allow_abbrev=False)
    runtime_parser.add_argument("--runtime", choices=["auto", "rust", "python"], default="auto")
    ns, args = runtime_parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    if ns.runtime == "python":
        return _run_python(args)
    rt, reason = _load_rust()
    if ns.runtime == "rust":
        if rt is None:
            sys.exit(f"{PROG}: {reason}")
        return _run_rust(rt, args)

    if "-h" in args or "--help" in args:
        print(HELP, flush=True)
        return _run_rust(rt, args) if rt is not None else _run_python(args, reason)
    if rt is None:
        return _run_python(_python_args(args), reason)
    try:
        rt.parse_args([PROG, *args])
    except ValueError as e:
        if not _python_accepts(args):  # neither runtime takes the arguments: report the default runtime's error
            sys.stderr.write(str(e))
            sys.exit(2)
        first = str(e).strip().splitlines()[0].removeprefix("error: ")
        return _run_python(args, f"only it takes these arguments ({first})")
    _run_rust(rt, args)


if __name__ == "__main__":
    main()
