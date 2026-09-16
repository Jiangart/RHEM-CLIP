
# RHEM-CLIP
Official implementation of **RHEM-CLIP: Reliability-Grounded
Hierarchical Evidence Modeling for Zero-Shot Structural Anomaly Detection**.

## Environment
conda create -n rhemclip python=3.10
conda activate rhemclip
pip install -r requirements.txt

## Training
bash train.sh

## Evaluation
bash test.sh

## Datasets
Experiments are conducted on MVTec AD, VisA, MPDD, SDD, BTAD,
DAGM, DTD-Synthetic, and WTB.
Please download the datasets from their official sources.
