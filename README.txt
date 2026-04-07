PaDT - README
=============

Project: PaDT (7B)
Version: 1.0

System Requirements
-------------------
- Storage: 20GB+ free disk space
- VRAM: 24GB (trained on a single NVIDIA RTX 3090 24GB)
- OS: Linux recommended
- Python: 3.11
- Package Manager: Conda

Installation
------------
1. Ensure you have Python 3.11 and Conda installed.
2. Run the installation script:
   ./install.sh
3. Activate the Conda environment:
   conda activate PaDT

Training
--------
1. Activate the environment:
   conda activate PaDT

2. (Optional) Edit training configuration:
   Modify cfg.json as needed

3. Start training:
   python3 src/train_padt/train.py

Note:
- On the first run, the base model will be automatically downloaded to:
  zoo/padt_7b_rec/
- Subsequent runs will reuse the downloaded model from this folder.

Evaluation
----------
1. Activate the environment:
   conda activate PaDT

2. Run evaluation:
   python3 src/train_padt/evaluate.py

Additional Notes
----------------
- Training was performed using a single NVIDIA RTX 3090 with 24GB VRAM.
- Make sure you have sufficient storage before downloading the model.
- All scripts must be run with the PaDT conda environment activated.

