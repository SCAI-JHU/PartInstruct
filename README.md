# PartInstruct

### Environment Setup 


```bash
git clone --recurse-submodules https://github.com/SCAI-JHU/PartInstruct.git
cd PartInstruct
conda env create -f part_instruct.yml
conda activate part_instruct
pip install -r requirements.txt
cd third_party/pybullet_planning/
pip install .
cd ../../
pip install -e .
```

### Download Assets
Go to the dataset page: https://huggingface.co/datasets/SCAI-JHU/PartInstruct. Log in to your Hugging Face account and accept the conditions as prompted. Then go back to the project root directory, log in from your terminal.

```bash
huggingface-cli login
```
Enter your password. You can now download the assets.

```bash
huggingface-cli download SCAI-JHU/PartInstruct --repo-type dataset --local-dir ./data --include "*.json"
huggingface-cli download SCAI-JHU/PartInstruct --repo-type dataset --local-dir ./data --include "assets.zip"
unzip ./data/assets.zip -d ./data/
mv data/assets/* data/
rm -r data/assets*
```

