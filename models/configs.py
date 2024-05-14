# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#TRANSFORMER CONFIGURATIONS
configurations = {}
mlp_dim = [2048,3072]
n_layers = [4,6,8]
hs_nh = [(64,4), (32,8), (16,16)]
k=1

for dim in mlp_dim:
    for n in n_layers:
        for hs,nh in hs_nh:
            configurations['Configuration '+str(k)] = [16,dim,n,hs,nh]
            k += 1
            
            
mlp_dim = [4096]
n_layers = [4,6,8]
hs_nh = [(256,4), (128,8), (64,16)]

k = 19
for dim in mlp_dim:
    for n in n_layers:
        for hs,nh in hs_nh:
            configurations['Configuration '+str(k)] = [32,dim,n,hs,nh]
            k += 1
                     

mlp_dim = [2204]
n_layers = [4,6]
hs_nh = [(16,4),(8,8)]

k = 28
for dim in mlp_dim:
    for n in n_layers:
        for hs,nh in hs_nh:
            configurations['Configuration '+str(k)] = [8,dim,n,hs,nh]
            k += 1
            

            
conf = 5 # configuration to be choose
ps,dim,n,hs,nh = configurations['Configuration '+str(conf)][0], configurations['Configuration '+str(conf)][1], configurations['Configuration '+str(conf)][2],  configurations['Configuration '+str(conf)][3],  configurations['Configuration '+str(conf)][4]


import ml_collections

def get_eva_config():
    """Returns the EvaViT configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (ps, ps, 5)})
    config.hidden_size = hs
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = dim
    config.transformer.num_heads = nh
    config.transformer.num_layers = n
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config
    

def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    del config.patches.size
    config.patches.grid = (14, 14)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config
