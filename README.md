# gender
This repo contains the code for training and inference of a model generating descriptions for clusters of genes.
# Setup 
```bash
pip3 install -r requirements.txt
```

# Inference
First, download a trained checkpoint from google drive (skip this if you use your own trained model for inference):
```python
import gdown
import os

os.makedirs('trained_models', exist_ok=True)

link = 'https://drive.google.com/file/d/1IqByKbVmA6crv30BjFwg5MX6nIyA6M9O/view?usp=sharing'
filename = 'large-perm20-neg0.05-desc_def'
gdown.download(url=link, output=f"trained_models/{filename}.zip", fuzzy=True)
os.system(f'unzip trained_models/{filename}.zip -d trained_models/')
os.system(f'rm trained_models/{filename}.zip')
```
Now, you can generate predictions for your test file using `inference.py`:
```bash
python3 inference.py --checkpoint_path trained_models/large-perm20-neg0.05-desc_def/checkpoint --test_file data/test.tsv -o path/to/output.txt
```
# Train
To train model from scratch run `train.py`:
```bash
python3 train.py \
-m microsoft/biogpt-large \
--dataset ['biocarta', 'kegg', 'wp', 'pid', 'reactome'] \
--data_dir data
```
See `python3 train.py --help` for full documentation
