"""
Remote detokenizer service entry point.

Run on the detokenizer node (single GPU).
"""

import argparse

from .remote_detokenizer import RemoteDetokenizerServer
from .utils import get_logger, set_global_log_level


def main():
    parser = argparse.ArgumentParser(description="Vox-Serve Remote Detokenizer Service")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--token-bind",
        type=str,
        default="tcp://0.0.0.0:5557",
        help="Bind address for incoming token chunks (default: tcp://0.0.0.0:5557)",
    )
    parser.add_argument(
        "--audio-bind",
        type=str,
        default="tcp://0.0.0.0:5558",
        help="Bind address for outgoing audio chunks (default: tcp://0.0.0.0:5558)",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument("--enable-torch-compile", action="store_true", help="Enable torch.compile")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)",
    )
    args = parser.parse_args()

    set_global_log_level(args.log_level)
    logger = get_logger(__name__)

    server = RemoteDetokenizerServer(
        model_name=args.model,
        token_bind=args.token_bind,
        audio_bind=args.audio_bind,
        device=args.device,
        enable_torch_compile=args.enable_torch_compile,
        logger=logger,
    )
    logger.info("Detokenizer service ready")
    server.run_forever()


if __name__ == "__main__":
    main()
