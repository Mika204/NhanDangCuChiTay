"""Entry point don gian cho demo anh va webcam."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Nhan dien cu chi tay - HOG + SVM")
    parser.add_argument("mode", choices=["test", "webcam"], nargs="?", default="test")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.mode == "webcam":
        from webcam_demo import run
        run(args.camera)
    else:
        from test_model import demo
        demo(show=not args.no_show)


if __name__ == "__main__":
    main()
