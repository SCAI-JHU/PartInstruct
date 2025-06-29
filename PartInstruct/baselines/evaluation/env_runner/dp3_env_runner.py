from PartInstruct.baselines.evaluation.env_runner.base_env_runner import BaseEnvRunner

class DP3EnvRunner(BaseEnvRunner):

    def get_obs(self, obs, past_action, device):
        obs_dict = super().get_obs(obs, past_action, device)
        return obs_dict