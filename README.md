# Golf Ball Broadcast Detection Model

This is a model capable of detecting golf balls on a PGA broadcast

## Dataset
- Due to the small bounding boxes of the golf ball, the images were sliced via SAHI to cut the images into sections
- The list of the youtube ids for the videos used to train this model have been included

## Model
- YOLOv8 Nano (3M parameters)

## Training
- Recall: 94%
- Precision: 96%
- Trained on an RTX 3080 10GB

## Experimental
- Optical Flow Estimation was used to counteract the camera centering the moving ball in the same position
- I provided an example snippet of uploading XML annotations 