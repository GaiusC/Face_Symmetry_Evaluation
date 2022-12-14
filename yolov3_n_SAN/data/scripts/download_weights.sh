#!/bin/bash
# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
# Download latest models from https://github.com/ultralytics/yolov3/releases
# Example usage: bash path/to/download_weights.sh
# parent
# └── yolov3_n_SAN
#     ├── yolov3_n_SAN.pt  ← downloads here
#     ├── yolov3_n_SAN-spp.pt
#     └── ...

python - <<EOF
from utils.downloads import attempt_download

models = ['yolov3', 'yolov3-spp', 'yolov3-tiny']
for x in models:
    attempt_download(f'{x}.pt')

EOF
