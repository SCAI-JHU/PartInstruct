import torch
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np

class T5Encoder(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path='t5-small'):
        super().__init__()
        self.t5_encoder = T5EncoderModel.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path)

    def tokenize(self, text: str) -> dict:
        output = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        output_np = {key: value.numpy() for key, value in output.items()}
        return output_np
    
    def encode_from_tokenized(self, tokenized: dict):
        outputs = self.t5_encoder(**tokenized)
        # Mean pooling across the sequence dimension (sequence length is the second dimension)
        sentence_embeddings = outputs.last_hidden_state.mean(dim=1)
        return sentence_embeddings

    def decode_tokenized(self, tokenized: dict) -> list:
        input_ids = tokenized['input_ids']
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.numpy()
        
        reshaped = False
        if len(input_ids.shape) == 3:
            # Reshape from [T, B, N] to [T*B, N]
            T, B, N = input_ids.shape
            input_ids = input_ids.reshape(T * B, N)
            reshaped = True
        
        decoded_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Reshape the decoded text back to [T, B]
        if reshaped:
            decoded_text = np.array(decoded_text).reshape(T, B).tolist()
        
        return np.array(decoded_text)

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        outputs = self.t5_encoder(**inputs)
        # Mean pooling across the sequence dimension (sequence length is the second dimension)
        sentence_embeddings = outputs.last_hidden_state.mean(dim=1)
        return sentence_embeddings

    def output_shape(self):
        hidden_size = self.t5_encoder.config.d_model
        return (hidden_size,) 