from PartInstruct.baselines.evaluation.env_runner.base_env_runner import BaseEnvRunner

class DPEnvRunner(BaseEnvRunner):

    def get_obs(self, obs, past_action, device):
        obs['agentview_rgb'] = obs['agentview_rgb'] / 255.0
        obs['agentview_mask'] = obs['agentview_part_mask']

        obs_dict = super().get_obs(obs, past_action, device)
        del obs_dict['agentview_part_mask']
        return obs_dict



