import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run either train.py or test.py"
    )
    parser.add_argument(
        "mode", type=str, choices=["train", "test"], help="Mode: train or test"
    )
    args, remaining_argv = parser.parse_known_args()

    module_map = {"train": "src.train", "test": "src.test"}

    module_name = module_map.get(args.mode)
    if not module_name:
        raise Exception(f"Invalid mode: {args.mode}")

    sys.argv = [sys.argv[0]] + remaining_argv
    module = __import__(module_name, fromlist=["run"])
    
    module.run()


if __name__ == "__main__":
    main()
