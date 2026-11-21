# YOLOX-S surround camera weights

The unmodified FP32 COCO YOLOX-S ONNX file comes from OpenCV Zoo commit
`48b10fa505aeb51fbd127fc60a5c660040e823b7`. The upstream directory licenses
its files under Apache License 2.0. This directory includes the upstream license.

- [OpenCV model card and license statement](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox)
- [Pinned Git LFS pointer with size and checksum](https://raw.githubusercontent.com/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox/object_detection_yolox_2022nov.onnx)
- [Original weights](https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox/object_detection_yolox_2022nov.onnx)

Size is 35,858,002 bytes. SHA-256 is
`c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`.
The observer uses two ONNX Runtime CPU threads when available and supports an OpenCV DNN CPU fallback. It uses the upstream RGB 0–255 input,
top-left origin, 114 padding, linear resizing, and stride-8/16/32 decoder.
Each box records the winning COCO class and objectness times class confidence.
