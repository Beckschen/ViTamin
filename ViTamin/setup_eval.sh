# env set up
conda create -n vitamin_eval python=3.9
conda activate vitamin_eval
pip3 install -r requirements-eval.txt # follow datacomp but remove some dependencies
pip3 install torch torchvision torchaudio
pip3 install timm open_clip_torch

git clone https://github.com/mlfoundations/datacomp.git
cd datacomp
ln -s ../open_clip open_clip
ln -s ../models models
cd ..

# download_dir='YOUR_EVAL_DATA_DIR'
# python download_evalsets.py $download_dir