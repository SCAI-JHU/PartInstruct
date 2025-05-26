# PartInstruct

### Environment Setup 


```bash
git clone --recurse-submodules https://github.com/SCAI-JHU/PartInstruct.git
cd PartInstruct
conda create -n partinstruct -c conda-forge python=3.9 cmake=3.24.3 open3d gxx_linux-64 ninja gcc_linux-64=12 gxx_linux-64=12
conda activate partinstruct
pip3 install torch torchvision torchaudio
```

Install third-party

```bash
pip install -r requirements.txt

pip install -e .
pip install -e ./third_party/pybullet_planning/
pip install -e ./third_party/diffusion_policy/
pip install -e ./third_party/gym-0.21.0/
pip install -e ./third_party/pytorch3d_simplified/
pip install -e ./third_party/sam_2/

```

### Download Assets and Pretrained Checkpoints
Go to the dataset page: https://huggingface.co/datasets/SCAI-JHU/PartInstruct. Log in to your Hugging Face account and accept the conditions as prompted. Then go back to the project root directory, log in from your terminal.

```bash
huggingface-cli login
```
Enter your password. You can now download the assets. The following commands download and set up the assets under a created data/ directory.

```bash
huggingface-cli download SCAI-JHU/PartInstruct --repo-type dataset --local-dir ./data --include "*.json" "assets.zip" "checkpoints/**" 

#To download PartInstruct dataset in hdf5 format, add "demos/**" for all demo, "demos/OBJECT_NAME.hdf5" for demo of specific object type

unzip ./data/assets.zip -d ./data/ && mv data/assets/* data/
rm -r data/assets*
```

Download checkpoints of SAM-2 (Use in Bi-level Planning)

```bash
cd ./third_party/sam_2/checkpoints/
bash download_ckpts.sh
cd ../../../
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

### Run Evaluation

```bash
# e.g. Run Evaluation with Diffusion Policy (DP)
python PartInstruct/baselines/evaluation/evaluator.py \
    --config-name dp_evaluator_mask_one_enc \
    rollout_mode='specific_ckpt' \
    split='test1' \
    horizon=16 \
    n_obs_steps=2 \
    n_action_steps=8 \
    job_id='0' \
    ckpt_path=PartInstruct/data/checkpoints/diffusion_policy/latest.ckpt \
    output_dir=PartInstruct/outputs/ \
    task.env_runner.n_envs=1 \
    task.env_runner.n_vis=1
```

