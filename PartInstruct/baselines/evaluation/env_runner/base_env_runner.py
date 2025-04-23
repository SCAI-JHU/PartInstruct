import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import json
import os
import pickle
import importlib

import robomimic.utils.tensor_utils as TensorUtils
from PartInstruct.baselines.utils.encoders import T5Encoder
from PartInstruct.baselines.utils.async_vector_env import AsyncVectorEnv
from PartInstruct.baselines.utils.multistep_wrapper import MultiStepWrapper
from PartInstruct.baselines.utils.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from PartInstruct.baselines.utils.log_utils import append_to_info_file, combine_to_json
from PartInstruct.baselines.policy.base_policy import BasePolicy
from PartInstruct.baselines.utils.robodiff_pytorch_utils import dict_apply


class BaseEnvRunner():
    def __init__(self,
            output_dir,
            lang_encoder: T5Encoder,
            obj_class=None,
            start_seed=10000,
            max_steps=250,
            n_obs_steps=8,
            n_action_steps=8,
            fps=10,
            crf=22,
            env_config="config/3dp_config.yaml", 
            gui=False, 
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            n_vis=0,
            split=None,
            task_types=None,
            epoch=None,
            bullet_env=None,
            metrics_mode='default',
            log_mode='verbose'
        ):
        self.lang_encoder = lang_encoder
        self.log_mode = log_mode
        self.metrics_mode = metrics_mode
        self.output_dir = pathlib.Path(output_dir).joinpath(
                        'media', str(epoch), str(obj_class), 'split-'+ str(split))

        print("#### Evaluation results output to:", self.output_dir)

        steps_per_render = max(10 // fps, 1)
        module = importlib.import_module(bullet_env)
        BulletEnv = getattr(module, "BulletEnv")
        def env_fn(obj_class, split, task_type, max_steps):
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    BulletEnv(
                        obj_class=obj_class,
                        evaluation=True,
                        split=split,
                        task_type=task_type,
                        record=False,
                        config_path=env_config,
                        gui=gui
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [
            lambda obj_class=obj_class, split=split, task_type=task_types[i]: 
                env_fn(
                    obj_class, 
                    split, 
                    task_type, 
                    max_steps=400 if str(split) == 'test4' else 250
                ) 
            for i in range(n_envs)
        ]

        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        self.filename_info_path = pathlib.Path(self.output_dir).joinpath(
            'video_filename.info')
        self.task_info_path = pathlib.Path(self.output_dir).joinpath(
            'video_taskname.info')
        nth_env = 0

        for i in range(n_envs):
            nth_env += 1
            seed = start_seed + i
            enable_render = i < n_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filepath = pathlib.Path(self.output_dir).joinpath(
                        'task_type-' + str(task_types[i]) + '_env-' + str(nth_env) + ".mp4")
                    filepath.parent.mkdir(parents=False, exist_ok=True)
                    filepath = str(filepath)
                    filename = os.path.basename(filepath)
                    env.env.file_path = filepath
                    append_to_info_file(filename, self.filename_info_path)

                assert isinstance(env, MultiStepWrapper)
            
            env_seeds.append(seed)
            env_prefixs.append(split + '/' + str(task_types[i]) + '/' + 'env_' + str(i) + '/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def get_obs(self, obs, past_action, device):
        np_obs_dict = dict(obs)
        np_tokenized_language = {
            "input_ids": torch.from_numpy(np_obs_dict["input_ids"]),
            "attention_mask": torch.from_numpy(np_obs_dict["attention_mask"])
        }
        del np_obs_dict["attention_mask"]
        del np_obs_dict["input_ids"]

        np_instruction = TensorUtils.time_distributed(np_tokenized_language, self.lang_encoder.encode_from_tokenized).detach().numpy()
        np_obs_dict["instructions"] = np_instruction

        if self.past_action and (past_action is not None):
            np_obs_dict['past_action'] = past_action[
                :,-(self.n_obs_steps-1):].astype(np.float32)
        
        obs_dict = dict_apply(np_obs_dict, 
            lambda x: torch.from_numpy(x).to(
                device=device))
        
        return obs_dict

    def run(self, policy: BasePolicy):
        device = policy.device
        env = self.env
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_infos = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()

            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval BulletEnvRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            dump = True
            while not done:
                obs_dict = self.get_obs(obs=obs, past_action=past_action, device=device)
                
                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                    
                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                # step env
                obs, reward, done, info = env.step(action)
                if dump:
                    for i in range(len(info)):
                        chain_params = info[i]['chain_params'][0]
                        chain_params = str(chain_params)
                        append_to_info_file(chain_params, self.task_info_path)
                
                done = np.all(done)
                past_action = action

                pbar.update(action.shape[1])
                dump = False

            output_json_path = pathlib.Path(self.output_dir).joinpath('video_meta.json')
            log_json_path = pathlib.Path(self.output_dir).joinpath('rollout_logs.json')
            combine_to_json(self.task_info_path, self.filename_info_path, output_json_path)

            pbar.close()
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            all_infos[this_global_slice] = list(info)

            if os.path.exists(self.filename_info_path):
                os.remove(self.filename_info_path)
            if os.path.exists(self.task_info_path):
                os.remove(self.task_info_path)
        
        # clear out video buffer
        _ = env.reset()

        # Calculate and log metrics
        log_data, metrics, verbose_metrics = self.calculate_metrics(n_inits, all_rewards, all_infos, all_video_paths, self.metrics_mode)

        # save log data
        with open(log_json_path, 'w') as f:
            json.dump(log_data, f, indent=4)
        
        if self.log_mode == 'verbose':
            verbose_log_pkl_path = os.path.join(os.path.dirname(log_json_path), 'render_log.pkl')
            with open(verbose_log_pkl_path, 'wb') as f:
                pickle.dump(verbose_metrics, f)
        return log_data

    def calculate_metrics(self, n_inits, all_rewards, all_infos, all_video_paths, metrics_mode='default'):
        metrics = {
            'rewards': collections.defaultdict(list),
            'successes': collections.defaultdict(list),
            'completion_rates': collections.defaultdict(list),
            'steps': collections.defaultdict(list),
            'ep_id': collections.defaultdict(list)
        }

        if metrics_mode == 'sam':
            metrics.update({
                'grounding_success': collections.defaultdict(list),
                'iou': collections.defaultdict(list),
                'accuracy': collections.defaultdict(list)
            })

        verbose_metrics = {
            'action': collections.defaultdict(list),
            'obj_id': collections.defaultdict(list),
            'obj_class': collections.defaultdict(list),
            'task_type': collections.defaultdict(list),
            'obj_scale': collections.defaultdict(list),
            'obj_pos': collections.defaultdict(list),
            'obj_orient': collections.defaultdict(list),
            'chain_params': collections.defaultdict(list),
        }

        log_data = dict()
        for i in range(n_inits):
            prefix = self.env_prefixs[i]
            log_data.update({
                f'{prefix}max_reward_{i}': float(np.max(all_rewards[i])),
                f'{prefix}success_{i}': float(all_infos[i]['Success'][-1]),
                f'{prefix}completion_rate_{i}': float(all_infos[i]['Completion Rate'][-1]),
                f'{prefix}steps_{i}': int(all_infos[i]['Steps'][-1]),
                f'{prefix}ep_id_{i}': int(all_infos[i]["ep_id"][-1]),
            })

            if metrics_mode == 'sam':
                log_data.update({
                    f'{prefix}grounding_success_{i}': float(all_infos[i]['Grounding Success'][-1]),
                    f'{prefix}iou_{i}': float(np.mean(all_infos[i]['iou'])),
                    f'{prefix}accuracy_{i}': float(np.mean(all_infos[i]['accuracy'])),
                })

            metrics["ep_id"][prefix].append(int(all_infos[i]["ep_id"][-1]))
            metrics['rewards'][prefix].append(float(np.max(all_rewards[i])))
            metrics['successes'][prefix].append(float(all_infos[i]['Success'][-1]))
            metrics['completion_rates'][prefix].append(float(all_infos[i]['Completion Rate'][-1]))
            metrics['steps'][prefix].append(int(all_infos[i]['Steps'][-1]))

            if metrics_mode == 'sam':
                metrics['grounding_success'][prefix].append(float(all_infos[i]['Grounding Success'][-1]))
                metrics['iou'][prefix].append(float(np.mean(all_infos[i]['iou'])))
                metrics['accuracy'][prefix].append(float(np.mean(all_infos[i]['accuracy'])))

            verbose_metrics['action'][prefix].extend(all_infos[i]['Action'])
            verbose_metrics['obj_id'][prefix].extend(all_infos[i]['Object id'])
            verbose_metrics['obj_class'][prefix].extend(all_infos[i]['obj_class'])
            verbose_metrics['task_type'][prefix].extend(all_infos[i]['Task type'])
            verbose_metrics['obj_scale'][prefix].extend(all_infos[i]['Object scale'])
            verbose_metrics['obj_pos'][prefix].extend(all_infos[i]['Object init pose'])
            verbose_metrics['obj_orient'][prefix].extend(all_infos[i]['Object init orient'])
            verbose_metrics['chain_params'][prefix].extend(all_infos[i]['chain_params'])

            video_path = all_video_paths[i]
            if video_path is not None:
                wandb.Video(video_path)

        # Log aggregate metrics
        for metric_name, metric_dict in metrics.items():
            for prefix, values in metric_dict.items():
                filtered_values = [v for v in values if not np.isnan(v)]  # Remove NaN values (Only for SAM)
                if filtered_values:
                    log_data[prefix + f'avg_{metric_name}'] = int(np.mean(filtered_values)) if isinstance(filtered_values[0], int) else np.mean(filtered_values)

        return log_data, metrics, verbose_metrics
