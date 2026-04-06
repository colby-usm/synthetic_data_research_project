# Note, our model was trained with 1 NVIDA RTX 3090 24GB of VRAM with CUDA version --

# Installation
1. System Requirements: Python 3.11, Conda
2. run install.sh
3. activate conda env with conda activate PaDT

# Run Training Loop
1. Ensure conda env is active (conda activate PaDT)
2. navigate to src/train_padt
3. Make any changes to cfg.json
4. run training loop with python3 train.py

Note, the model will be downloaded to zoo/padt_7b_rec/ on the first run. Subsequent runs will use this downloaded model

# Run Evaluation Loop
1. Add the model path to the eval dictionary under cfg.json
2. Navigate to src/train_padt
2. Run the eval loop with  python3 evaluate.py
