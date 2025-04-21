# Realtime SAM2.1 + Grounding DINO

To install and compile, simply run
```bash
./install.sh
```

To use SAM2.1 + Grounding Dino, users can use the `GroundedSAM2Predictor` class as shown below.
A demo can be run by running `grounded_sam2/inference.py`.
```python
import cv2
from grounded_sam2.inference import GroundedSAM2Predictor

gsam_predictor = GroundedSAM2Predictor()

cap = cv2.VideoCapture("your_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Language commands should be in the format of "class1, class2, class3"
    language = "yellow circle, light blue square, red square"

    masks = gsam_predictor.query(frame, language, display=(938, 532))

    key = "yellow circle"
    cv2.imshow(key, masks[key])
    cv2.waitKey(1)

cap.release()
```
