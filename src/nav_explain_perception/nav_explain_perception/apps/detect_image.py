"""``nep_detect`` — run YOLO on a still image and print detections."""

from __future__ import annotations

import argparse
import sys

from ..detector import YoloDetector, torch_available


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Detect objects in an image with YOLO.")
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="yolov5s")
    p.add_argument("--conf", type=float, default=0.4)
    args = p.parse_args(argv)

    if not torch_available():
        print("torch is not available in this interpreter; install it to run detection.")
        return 0
    import cv2

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read image: {args.image}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = YoloDetector(args.model, args.conf).detect(rgb)
    print(f"{len(dets)} detections:")
    for d in dets:
        print(f"  {d.label:15s} conf={d.confidence:.2f} box={tuple(round(v, 1) for v in d.box)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
