import argparse
import sys


def main():
    # create the ArgumentParser object
    parser = argparse.ArgumentParser(
        description="Run either train.py or test.py"
    )

    parser.add_argument(
        "--gui",
        default=False,
        action="store_true",
        help="If set, the GUI for the mode will also be parsed",
    )

    parser.add_argument(
        "mode",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Mode: train or test",
    )

    args, remaining_argv = parser.parse_known_args()

    if len(sys.argv) == 2:
        # User only provided mode argument, show help for corresponding module
        if args.mode == "train":
            from src.train import run
        else:
            from src.test import run
        run("--help")
    else:
        if args.gui:
            module_map = {
                "train": "src.gui.train_gui",
                "test": "src.gui.test_gui",
            }
        else:
            module_map = {"train": "src.train", "test": "src.test"}

        module_name = module_map.get(args.mode)
        if not module_name:
            raise Exception(f"Invalid mode: {args.mode}")

        sys.argv = [sys.argv[0]] + remaining_argv
        module = __import__(module_name, fromlist=["run"])

        module.run()


if __name__ == "__main__":
    main()
