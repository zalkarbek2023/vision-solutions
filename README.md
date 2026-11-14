# Vision solutions

Small, measured computer vision solutions on NVIDIA GPUs with Ultralytics YOLO and TensorRT, mostly for endoscopy.
Each folder has its code, its numbers on data the model never saw, and its honest limits.

| # | Solution | Tools | Headline number |
|---|---|---|---|
| 01 | [Erase burned-in patient data from endoscopy video, on the fly](01-deid/) | Ultralytics YOLO26-Text, NVIDIA TensorRT FP16 | 99.88 % of 22,440 words, 7.1 ms per frame |

Research prototypes, not medical devices. Code: AGPL-3.0 (it builds on Ultralytics).

Zalkarbek Arapbaev, physician and computer vision engineer.
