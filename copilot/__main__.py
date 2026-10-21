"""Chat with the copilot in the terminal:  python -m copilot"""

from .agent import Copilot
from .planner import get_planner


def main() -> None:
    planner = get_planner()
    copilot = Copilot(planner)
    print(f"Banking support copilot ({type(planner).__name__}). Signed in as demo customer C001. Ctrl+C to quit.\n")
    thread = None
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        res = copilot.chat("C001", text, thread)
        thread = res["thread_id"]
        while res["pending_approval"]:
            for action in res["pending_approval"]:
                print(f"copilot> I'm about to run {action['tool']} with {action['args']}.")
            ok = input("approve? [y/N] ").strip().lower() == "y"
            res = copilot.respond_to_approval(thread, ok)
        print(f"copilot> {res['reply']}\n")


if __name__ == "__main__":
    main()
