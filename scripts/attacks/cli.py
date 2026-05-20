from __future__ import annotations

import argparse
from pathlib import Path

from .common import save_reconstruction_result
from .dlg import DLGConfig, run_dlg
from .idlg import IDLGConfig, run_idlg
from .tableak import TabLeakConfig, run_tableak


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a phase-3 attack on a captured FL update.")
    parser.add_argument("--artifact", required=True, help="Path to the captured update artifact")
    parser.add_argument("--attack", required=True, choices=["dlg", "idlg", "tableak"])
    parser.add_argument("--processed-root", default=None, help="Optional processed dataset root")
    parser.add_argument("--output", default=None, help="Optional output path for the reconstruction")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--tau-start", type=float, default=1.0)
    parser.add_argument("--tau-end", type=float, default=0.01)
    args = parser.parse_args()

    if args.attack == "dlg":
        result = run_dlg(
            captured_update_path=args.artifact,
            processed_root=args.processed_root,
            config=DLGConfig(
                steps=args.steps or 3000,
                learning_rate=args.lr,
                device=args.device,
            ),
        )
    elif args.attack == "idlg":
        result = run_idlg(
            captured_update_path=args.artifact,
            processed_root=args.processed_root,
            config=IDLGConfig(
                steps=args.steps or 3000,
                learning_rate=args.lr,
                device=args.device,
            ),
        )
    else:
        result = run_tableak(
            captured_update_path=args.artifact,
            processed_root=args.processed_root,
            config=TabLeakConfig(
                steps=args.steps or 1500,
                learning_rate=args.lr,
                ensemble_restarts=args.restarts or 32,
                temperature_start=args.tau_start,
                temperature_end=args.tau_end,
                device=args.device,
            ),
        )

    output_path = Path(args.output) if args.output else Path(args.artifact).with_name(
        f"{args.attack}_reconstruction.pt"
    )
    saved_path = save_reconstruction_result(result, output_path)
    print(saved_path)


if __name__ == "__main__":
    main()
