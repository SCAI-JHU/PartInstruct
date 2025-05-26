
split='test5'
rollout_mode='specific_ckpt'

python PartInstruct/baselines/evaluation/evaluator.py \
    --config-name dp_evaluator_mask_one_enc \
    rollout_mode=${rollout_mode} \
    horizon=16 \
    n_obs_steps=2 \
    n_action_steps=8 \
    job_id='1111' \
    ckpt_path="PartInstruct/data/checkpoints/diffusion_policy/latest.ckpt" \
    output_dir=PartInstruct/outputs/ \
    task.env_runner.n_envs=1 \
    task.env_runner.n_vis=1
