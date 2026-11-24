"""``nep_detect``: run YOLO on a still image and print the detections.

Example::

    nep_detect --image photo.jpg --conf 0.5

Needs PyTorch, which is not a declared dependency. Without it the command says
so and exits with status 1.
"""

from __future__ import annotations

import argparse
import sys

from ..detector import DEFAULT_CONFIDENCE, DEFAULT_MODEL, YoloDetector, torch_available


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Detect objects in an image with YOLO.")
    p.add_argument("--image", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE)
    args = p.parse_args(argv)

    if not torch_available():
        print("torch is not available in this interpreter; install it to run detection.")
        return 1
    import cv2  # imported late, so that the missing-torch message needs no OpenCV

    img = cv2.imread(args.image)
    if img is None:
        print(f"could not read image: {args.image}")
        return 1
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = YoloDetector(args.model, args.conf).detect(rgb)
    print(f"{len(dets)} detections:")
    for d in dets:
        print(f"  {d.label:15s} conf={d.confidence:.2f} box={tuple(round(v, 1) for v in d.box)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
