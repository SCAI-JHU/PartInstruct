import numpy as np
from typing import Dict, Tuple, Union
import copy
from torch import nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from transformers import BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer
from PartInstruct.baselines.utils.robodiff_pytorch_utils import replace_submodules
from PartInstruct.baselines.utils.modules import *

####################
# Language Encoder #
####################

class BertEncoder(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = self.bert(**inputs)
        # Take the [CLS] token's embedding for each example in the batch
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings

    def output_shape(self):
        hidden_size = self.bert.config.hidden_size
        return (hidden_size,)

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


#################
# Image Encoder #
#################

class VisLangObsEncoder(nn.Module):
    def __init__(self,
            shape_meta: dict,
            rgb_encoder: Union[nn.Module, Dict[str,nn.Module]],
            lang_encoder: nn.Module, # same encoder for all language inputs, only for shape purposes
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_encoder: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        Assumes lang input: B,De
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        lang_keys = list()

        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_rgb_encoder:
            assert isinstance(rgb_encoder, nn.Module)
            key_model_map['rgb'] = rgb_encoder

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_encoder:
                    if isinstance(rgb_encoder, dict):
                        # have provided model for each key
                        this_model = rgb_encoder[key]
                    else:
                        assert isinstance(rgb_encoder, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_encoder)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                low_dim_keys.append(key)
            elif type == 'text':
                lang_keys.append(key)
                key_model_map[key] = lang_encoder
                key_shape_map[key] = key_model_map[key].output_shape()
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        lang_keys = sorted(lang_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_encoder = share_rgb_encoder
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.lang_keys = lang_keys
        self.key_shape_map = key_shape_map

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def forward(self, obs_dict):
        # print("!!!! obs_dict", obs_dict)
        batch_size = None
        features = list()
        # process rgb input
        if self.share_rgb_encoder:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)
        # import ipdb; ipdb.set_trace()
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            # import ipdb; ipdb.set_trace()
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)
        # import ipdb; ipdb.set_trace()
        # process language input
        for key in self.lang_keys:
            # print("key", key)
            data = obs_dict[key]
            # print("data", data)
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            # print("feature", feature)
            # import ipdb; ipdb.set_trace()
            features.append(data)

        # concatenate all features
        result = torch.cat(features, dim=-1)
        # import ipdb; ipdb.set_trace()
        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_input_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_input_dict[key] = this_obs
        
        example_output = self.forward(example_input_dict)
        output_shape = example_output.shape[1:]
        return output_shape

class VisLangObsMaskEncoder(nn.Module):
    def __init__(self,
            shape_meta: dict,
            rgb_encoder: Union[nn.Module, Dict[str,nn.Module]],
            mask_encoder: Union[nn.Module, Dict[str,nn.Module]],
            lang_encoder: nn.Module, # same encoder for all language inputs, only for shape purposes
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_encoder: bool=False,
            # use single mask model for all mask inputs
            share_mask_encoder: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False
        ):
        """
        Assumes rgb input: B,C,H,W
        Assume mask input: B,C,H,W
        Assumes low_dim input: B,D
        Assumes lang input: B,De
        """
        super().__init__()

        rgb_keys = list()
        mask_keys = list()
        low_dim_keys = list()
        lang_keys = list()

        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_rgb_encoder:
            assert isinstance(rgb_encoder, nn.Module)
            key_model_map['rgb'] = rgb_encoder

        if share_mask_encoder:
            assert isinstance(mask_encoder, nn.Module)
            key_model_map['mask'] = mask_encoder

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_encoder:
                    if isinstance(rgb_encoder, dict):
                        # have provided model for each key
                        this_model = rgb_encoder[key]
                    else:
                        assert isinstance(rgb_encoder, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_encoder)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'mask':
                mask_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_mask_encoder:
                    if isinstance(mask_encoder, dict):
                        # have provided model for each key
                        this_model = mask_encoder[key]
                    else:
                        assert isinstance(mask_encoder, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(mask_encoder)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
    
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                low_dim_keys.append(key)
            elif type == 'text':
                lang_keys.append(key)
                key_model_map[key] = lang_encoder
                key_shape_map[key] = key_model_map[key].output_shape()
            elif type == 'pcd':
                continue
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
            
        rgb_keys = sorted(rgb_keys)
        mask_keys = sorted(mask_keys)
        low_dim_keys = sorted(low_dim_keys)
        lang_keys = sorted(lang_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_encoder = share_rgb_encoder
        self.share_mask_encoder = share_mask_encoder
        self.rgb_keys = rgb_keys
        self.mask_keys = mask_keys
        self.low_dim_keys = low_dim_keys
        self.lang_keys = lang_keys
        self.key_shape_map = key_shape_map

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def forward(self, obs_dict):
        # print("!!!! obs_dict", obs_dict)
        batch_size = None
        features = list()
        # process rgb input
        if self.share_rgb_encoder:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)

        # process mask input
        if self.share_mask_encoder:
            # pass all mask obs to mask model
            masks = list()
            for key in self.mask_keys:
                mask = obs_dict[key]
                if batch_size is None:
                    batch_size = mask.shape[0]
                else:
                    assert batch_size == mask.shape[0]
                assert mask.shape[1:] == self.key_shape_map[key]
                mask = self.key_transform_map[key](mask)
                masks.append(mask)
            # (N*B,C,H,W)
            masks = torch.cat(masks, dim=0)
            # (N*B,D)
            feature = self.key_model_map['mask'](masks)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            features.append(feature)
        else:
            # run each mask obs to independent models
            for key in self.mask_keys:
                mask = obs_dict[key]
                # mask =  mask.unsqueeze(2)
                # import ipdb; ipdb.set_trace()
                if batch_size is None:
                    batch_size = mask.shape[0]
                else:
                    assert batch_size == mask.shape[0]
                assert mask.shape[1:] == self.key_shape_map[key]
                mask = self.key_transform_map[key](mask)
                feature = self.key_model_map[key](mask)
                features.append(feature)

        # import ipdb; ipdb.set_trace()
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            # import ipdb; ipdb.set_trace()
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)
        # import ipdb; ipdb.set_trace()
        # process language input
        for key in self.lang_keys:
            # print("key", key)
            data = obs_dict[key]
            # print("data", data)
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            # print("feature", feature)
            # import ipdb; ipdb.set_trace()
            features.append(data)

        # concatenate all features
        result = torch.cat(features, dim=-1)
        # import ipdb; ipdb.set_trace()
        return result

    @torch.no_grad()
    def output_shape(self):
        example_input_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_input_dict[key] = this_obs
        
        example_output = self.forward(example_input_dict)
        output_shape = example_output.shape[1:]
        return output_shape

class VisLangObsImageEncoder(nn.Module):
    def __init__(self,
            shape_meta: dict,
            image_encoder: Union[nn.Module, Dict[str,nn.Module]],
            lang_encoder: nn.Module, # same encoder for all language inputs, only for shape purposes
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single image model for all image inputs
            share_image_encoder: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        Assumes lang input: B,De
        """
        super().__init__()

        rgb_keys = list()
        mask_keys = list()
        low_dim_keys = list()
        lang_keys = list()

        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_image_encoder:
            assert isinstance(image_encoder, nn.Module)
            key_model_map['rgb'] = image_encoder
            key_model_map['mask'] = image_encoder

        obs_shape_meta = shape_meta['obs']
        value_to_replace = obs_shape_meta['agentview_part_mask']
        obs_shape_meta['agentview_mask'] = value_to_replace
        del obs_shape_meta['agentview_part_mask']
        # import ipdb; ipdb.set_trace()
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                key_prefix = key.split('_')[0]
                paired_mask_key = [k for k in obs_shape_meta.keys() if k.startswith(key_prefix) and k.endswith('mask')][0]
                # import ipdb; ipdb.set_trace()
                rgb_keys.append(key)
                mask_keys.append(paired_mask_key)
                # configure model for this key
                this_model = None
                if not share_image_encoder:
                    if isinstance(image_encoder, dict):
                        # have provided model for each key
                        this_model = image_encoder[key]
                    else:
                        assert isinstance(image_encoder, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(image_encoder)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406, 0.5, 0.5], std=[0.229, 0.224, 0.225, 0.225, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'mask':
                continue
            elif type == 'low_dim':
                low_dim_keys.append(key)
            elif type == 'text':
                lang_keys.append(key)
                key_model_map[key] = lang_encoder
                key_shape_map[key] = key_model_map[key].output_shape()
            elif type == 'pcd':
                continue
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
            
        rgb_keys = sorted(rgb_keys)
        mask_keys = sorted(mask_keys)
        low_dim_keys = sorted(low_dim_keys)
        lang_keys = sorted(lang_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_image_encoder = share_image_encoder
        self.rgb_keys = rgb_keys
        self.mask_keys = mask_keys
        self.low_dim_keys = low_dim_keys
        self.lang_keys = lang_keys
        self.key_shape_map = key_shape_map

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def forward(self, obs_dict):
        # print("!!!! obs_dict", obs_dict)
        batch_size = None
        features = list()
        # process rgb input
        if self.share_image_encoder:
            # stack rgb and paired mask and pass all image obs to image model
            imgs = list()
            for key in self.rgb_keys:
                key_prefix = key.split('_')[0]
                paired_mask_key = [k for k in self.mask_keys if k.startswith(key_prefix)][0]
                rgb = obs_dict[key]
                mask = obs_dict[paired_mask_key]
                # stack rgb and mask
                masks = mask.repeat(1, 2, 1, 1)
                img = torch.cat([rgb, masks], dim=1)
                # import ipdb; ipdb.set_trace()
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert rgb.shape[1:] == self.key_shape_map[key]
                assert mask.shape[1:] == self.key_shape_map[paired_mask_key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                key_prefix = key.split('_')[0]
                paired_mask_key = [k for k in self.mask_keys if k.startswith(key_prefix)][0]
                rgb = obs_dict[key]
                mask = obs_dict[paired_mask_key]
                # stack rgb and mask
                masks = mask.repeat(1, 2, 1, 1)
                img = torch.cat([rgb, masks], dim=1)
                # import ipdb; ipdb.set_trace()
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert rgb.shape[1:] == self.key_shape_map[key]
                assert mask.shape[1:] == self.key_shape_map[paired_mask_key]
                img = self.key_transform_map[key](img)
                # import ipdb; ipdb.set_trace()
                feature = self.key_model_map[key](img)
                features.append(feature)
        # import ipdb; ipdb.set_trace()
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            # import ipdb; ipdb.set_trace()
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)
        # import ipdb; ipdb.set_trace()
        # process language input
        for key in self.lang_keys:
            # print("key", key)
            data = obs_dict[key]
            # print("data", data)
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            # print("feature", feature)
            # import ipdb; ipdb.set_trace()
            features.append(data)

        # concatenate all features
        result = torch.cat(features, dim=-1)
        # import ipdb; ipdb.set_trace()
        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_input_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_input_dict[key] = this_obs
        
        example_output = self.forward(example_input_dict)
        output_shape = example_output.shape[1:]
        return output_shape

class FiLMLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gamma_transform = nn.Linear(in_channels, out_channels)
        self.beta_transform = nn.Linear(in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm2d(out_channels, affine=True)

        # Initialize weights and biases
        nn.init.xavier_uniform_(self.gamma_transform.weight)
        nn.init.zeros_(self.gamma_transform.bias)
        nn.init.xavier_uniform_(self.beta_transform.weight)
        nn.init.zeros_(self.beta_transform.bias)

    def forward(self, image_features, other_image_features):
        gamma = self.gamma_transform(other_image_features).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_transform(other_image_features).unsqueeze(-1).unsqueeze(-1)
        normalized_features = self.instance_norm(image_features)
        return (1 + gamma) * normalized_features + beta

class EfficientNetWithFiLM(torch.nn.Module):
    def __init__(self, language_feature_dim=768, model_name='efficientnet-b3', weights_path=None):
        super().__init__()
        self.resize = transforms.Resize((300, 300))
        self.efficient_net = EfficientNet.from_pretrained(model_name, weights_path)
        self.film_layers = torch.nn.ModuleList()

        for _, block in enumerate(self.efficient_net._blocks):
            num_features = block._block_args.input_filters
            self.film_layers.append(FiLMLayer(language_feature_dim, num_features))

    def forward(self, x, lang_features):
        if x.size(2) != 300 or x.size(3) != 300:
            x = torch.stack([self.resize(image) for image in x])

        x = self.efficient_net._swish(self.efficient_net._bn0(self.efficient_net._conv_stem(x)))
        # Blocks
        for idx, block in enumerate(self.efficient_net._blocks):
            film_layer = self.film_layers[idx]
            x = film_layer(x, lang_features)
            drop_connect_rate = self.efficient_net._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.efficient_net._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self.efficient_net._swish(self.efficient_net._bn1(self.efficient_net._conv_head(x)))
        
        x = self.efficient_net._avg_pooling(x)
        if self.efficient_net._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self.efficient_net._dropout(x)
            x = self.efficient_net._fc(x)
        
        return x

    def output_shape(self, input_shape):
        # Compute the output with dummy inputs
        dummy_input = torch.zeros(1, *input_shape)
        x = self.efficient_net.forward(dummy_input)
        return x.shape[1:]

def get_resnet(name, weights=None, input_channels=3, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    input_channels: Number of input channels (1 for grayscale, 3 for RGB, 4 for additional data)
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    
    # Modify the first conv layer for one-channel or four-channel input
    if input_channels == 1:
        resnet.conv1 = nn.Conv2d(1, resnet.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Average the pretrained weights across the 3 channels
            resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.mean(dim=1, keepdim=True))

    elif input_channels == 4:
        resnet.conv1 = nn.Conv2d(4, resnet.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Initialize the weights for the 4th channel as the average of the first 3 channels
            resnet.conv1.weight = nn.Parameter(
                torch.cat([resnet.conv1.weight, resnet.conv1.weight.mean(dim=1, keepdim=True)], dim=1)
            )

    # Remove the fully connected layer (for feature extraction)
    resnet.fc = torch.nn.Identity()
    
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model