"""User-facing commands for ADO — Autonomous Driver Omniverse."""
from importlib import import_module
import sys


COMMANDS = {
    "drive": "drive", "study": "experiment", "showcase": "showcase",
    "render-benchmark": "render_benchmark", "traction": "traction",
    "commission": "commission", "report": "report",
}


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("--help", "-h"):
        print("ADO — Autonomous Driver Omniverse\n")
        print("Use python -m ado <command> --help for options.\n")
        print("Commands: " + ", ".join(COMMANDS))
        return
    if args[0] not in COMMANDS:
        raise SystemExit(f"Unknown ADO command {args[0]!r}. Use python -m ado --help.")
    sys.argv = [f"ado {args[0]}", *args[1:]]
    import_module(f"crosswalk.{COMMANDS[args[0]]}").main()


if __name__ == "__main__":
    main()
