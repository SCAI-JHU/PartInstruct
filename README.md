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
huggingface-cli download SCAI-JHU/PartInstruct --repo-type dataset --local-dir ./data --include "*.json" "assets.zip"
unzip ./data/assets.zip -d ./data/
mv data/assets/* data/
rm -r data/assets*
```

### Run Demos with Oracle Policy
This command will sample part-level manipulation tasks from the evaluation metadata and execute the tasks using an Oracle planner:
```bash
python scripts/run_oracle_policy.py
```

### Evaluate Code as Policies
To evaluate [Code as Policies](https://code-as-policies.github.io/) with GPT4o, use the following command to set up your OpenAI API key as an environmental variable:
```bash
export OPENAI_API_KEY=your_openai_api_key
```
Then run the following command:
```bash
python scripts/run_code_as_policies.py
```

### Run evalutaion of a baseline

```
./scripts/eval_dp_mask.sh
```

