
split='test5'
rollout_mode='specific_ckpt'

python /home/jimmyhan/Desktop/lgplm/PartInstruct/Final_release/PartInstruct/PartInstruct/baselines/evaluation/evaluator.py \
    --config-name dp_evaluator_mask_one_enc \
    rollout_mode=${rollout_mode} \
    horizon=16 \
    n_obs_steps=2 \
    n_action_steps=8 \
    job_id='1111' \
    ckpt_path="/home/jimmyhan/Desktop/lgplm/PartInstruct/ckpt/dp-new-augumented_data_mask-09-24_09-41-16/latest.ckpt" \
    output_dir=/home/jimmyhan/Desktop/lgplm/PartInstruct/baselines/TEST_output \
    task.env_runner.n_envs=1 \
    task.env_runner.n_vis=1
