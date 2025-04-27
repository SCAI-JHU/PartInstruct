
split='TEST_SPLIT'
obj_classes='OBJ_CLASS'
task_type='TASK_TYPE'
rollout_mode='specific_ckpt'

python /PartInstruct/baselines/evaluation/evaluator.py \
    --config-name dp_evaluator_mask_one_enc \
    rollout_mode=${rollout_mode} \
    obj_classes=${obj_classes} \
    task_type=${task_type} \
    split=${split} \
    horizon=16 \
    n_obs_steps=2 \
    n_action_steps=8 \
    ckpt_path="path/to/ckpt" \
    output_dir="path/to/output" \
    task.env_runner.n_envs=1 \
    task.env_runner.n_vis=1
