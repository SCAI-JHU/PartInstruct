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
