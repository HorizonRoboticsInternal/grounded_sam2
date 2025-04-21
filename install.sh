#!/bin/bash

set -e

# First, download all checkpoints
cd sam2_checkpoints
bash download_ckpts.sh
cd ..
cd gdino_checkpoints
bash download_ckpts.sh
cd ..

# Then install sam2
pip install -e .

# After install gdino
cd grounding_dino
pip install -r requirements.txt
pip install -e .
