# Medical-Image-Processing
The Covid Detection application

# Installation Guide
Download Conda -> [source](https://repo.anaconda.com/archive/Anaconda3-2022.05-Windows-x86_64.exe)
## commands - 

### create a new env
```bash
conda create --prefix ./env python=3.7 -y
```

### activate new env
```bash
conda activate ./env
```
### install Packages 
```bash
pip install -r requirements.txt
```
### init DVC
```bash
git init
dvc init
```

### create empty files - 
```bash
mkdir -p src/utils config
touch src/__init__.py src/utils/__init__.py param.yaml dvc.yaml config/config.yaml src/stage_01_load_save.py src/utils/all_utils.py setup.py .gitignore
```

### install src 
```bash
pip install -e .
```