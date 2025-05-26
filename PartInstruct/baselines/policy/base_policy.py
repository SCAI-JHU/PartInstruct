from torch import nn

class BasePolicy(nn.Module):
    def __init__(self):
        super().__init__()

    def process_input_for_training(self, x):
        raise NotImplementedError

    def process_input_for_evaluation(self, x):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device    
    
    def reset(self):
        pass
