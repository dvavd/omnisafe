# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""This module contains the helper functions for the model."""

from __future__ import annotations

import numpy as np
from torch import nn

from omnisafe.typing import Activation, InitFunction


def initialize_layer(init_function: InitFunction, layer: nn.Linear) -> None:
    """Initialize the layer with the given initialization function.

    The ``init_function`` can be chosen from: ``kaiming_uniform``, ``xavier_normal``, ``glorot``,
    ``xavier_uniform``, ``orthogonal``.

    Args:
        init_function (InitFunction): The initialization function.
        layer (nn.Linear): The layer to be initialized.
    """
    if init_function == 'kaiming_uniform':
        nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
    elif init_function == 'xavier_normal':
        nn.init.xavier_normal_(layer.weight)
    elif init_function in ['glorot', 'xavier_uniform']:
        nn.init.xavier_uniform_(layer.weight)
    elif init_function == 'orthogonal':
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
    else:
        raise TypeError(f'Invalid initialization function: {init_function}')


def get_activation(
    activation: Activation,
) -> type[nn.Identity | nn.ReLU | nn.Sigmoid | nn.Softplus | nn.Tanh]:
    """Get the activation function.

    The ``activation`` can be chosen from: ``identity``, ``relu``, ``sigmoid``, ``softplus``,
    ``tanh``.

    Args:
        activation (Activation): The activation function.

    Returns:
        The activation function, ranging from ``nn.Identity``, ``nn.ReLU``, ``nn.Sigmoid``,
        ``nn.Softplus`` to ``nn.Tanh``.
    """
    activations = {
        'identity': nn.Identity,
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'softplus': nn.Softplus,
        'tanh': nn.Tanh,
    }
    assert activation in activations
    return activations[activation]


def _build_mlp_network(
    sizes: list[int],
    activation: Activation,
    output_activation: Activation = 'identity',
    weight_initialization_mode: InitFunction = 'kaiming_uniform',
) -> nn.Module:
    """Build the MLP network.

    Examples:
        >>> build_mlp_network([64, 64, 64], 'relu', 'tanh')
        Sequential(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ReLU()
            (4): Linear(in_features=64, out_features=64, bias=True)
            (5): Tanh()
        )

    Args:
        sizes (list of int): The sizes of the layers.
        activation (Activation): The activation function.
        output_activation (Activation, optional): The output activation function. Defaults to
            ``identity``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.

    Returns:
        The MLP network.
    """
    activation_fn = get_activation(activation)
    output_activation_fn = get_activation(output_activation)
    layers = []
    for j in range(len(sizes) - 1):
        act_fn = activation_fn if j < len(sizes) - 2 else output_activation_fn
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        initialize_layer(weight_initialization_mode, affine_layer)
        layers += [affine_layer, act_fn()]
    return nn.Sequential(*layers)

def build_mlp_network(
    sizes: list[int],
    activation: Activation,
    output_activation: Activation = 'identity',
    weight_initialization_mode: InitFunction = 'kaiming_uniform',
) -> nn.Module:
    return SocialNavHumanInteraction(sizes)


import pdb
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import time
"""
SocialNavHumanInteraction

- Lidar 1D CNN Based on the implementation of: https://journals.sagepub.com/doi/pdf/10.1177/0278364920916531
- Human Interaction LSTM based on the implementation of: https://github.com/vita-epfl/CrowdNav/tree/master/crowd_nav/policy


Input: 
    - Vector of lidar (goes to 1D Conv)
    - Current action (v, w)
    - Goal (dist, theta) or (dist_x, dist_y)
    - Human pose: a list of (px,py, vx, vy) robot-centric and map coordiantes for occupancy map
    
    
    
    
Architecture ():
1. [Lidar] -> 1DConv -> 1DConv -> Lidar_Linear

2. human_pose -> lstm -> last hidden

3

3. cat_linear (Lidar_Linear + robot_state + mlp3_output -> FC1 -> FC2 -> Action

RuntimeError: normal expects all elements of std >= 0.0 (this means graident exploding problem) So decrease grad_norm

Note: since this network uses a for-loop it's slow. To make it run faster:
1. Disable occupancy map (self.with_om = False)
2. Reduce the num_env (128), and horizon_length (64). 
3. minibatch_size (512)
"""


def compute_1dconv_size(L_in=50, kernel=3, stride=2, padding=0):
    """Computes the output size of 1d Conv"""
    return int((L_in + 2 * padding - (kernel-1) - 1)/stride + 1)  # int acts as floor function


class SocialNavHumanInteraction(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        actions_dim = 2
        self.kinematics = "holonomic"
        
        # hard-coded, must be same as task. Alternatively must define a dict
        self.lidar_hist_steps = 1  # lidar time-steps
        self.lidar_num_rays = 40  # TODO: fill
        self.humans_obs_limit = 10  # max num of observable humans  # TODO: fill
        self.robot_state_shape = 4  # for agent (vx,vy,gx,gy)
        self.human_state_shape = 4  # humans (px,py, vx,vy)
        ### Options ###
        self.add_lidar = True
        self.fixed_sigma = False
        self.learn_no_human_embedding = True  # set to True
        self.use_softmax_attn = True  # original code is False  ### TODO: Was True for trained polocy <<<<<<<<<<
        self.with_om = False  # be default False (for RLGames) but set to True for Imitation
        self.with_global_state = True  # add (concat) global_state (mean human states) to each human state between MLP1 and MLP2
        ###
        ### Occup Map
        print("Network with_om:", self.with_om)
        self.cell_size = 1.
        self.cell_num = 4
        self.om_channel_size = 3
        self.om_dim = self.cell_num ** 2 * self.om_channel_size
        ###
        human_input_dim = (self.human_state_shape + self.robot_state_shape) + (self.om_dim if self.with_om else 0)  # input_dim
        mlp1_dims = [150, 100]
        mlp2_dims = [100, 50]
        attention_dims = [100, 100, 1]  # 1 at the end to get score
        mlp3_dims = [150, 100, 100]  # , 1]  # 1 at the end commented to get more dense output
        self.human_out_dim = mlp3_dims[-1]
        #self.self_state_dim = self.robot_state_shape

        # 1DConv (Lidar)
        kernel1 = 5
        kernel2 = 3
        stride = 2
        channel1 = 32
        channel2 = 32
        lidar_out = 128 if self.add_lidar else 0
        # output FC
        FC1 = 256
        FC2 = 256
        ########

        self.is_value_function = sizes[-1] == 1

        print("SocialNavHumanInteraction initialised.")
        
        # stable-baseline
        self.stable_baseline = False # when called by stable-baseline (for imitation learining) since output is different
        self.latent_dim_pi = self.latent_dim_vf = self.robot_state_shape + lidar_out + self.human_out_dim

        if self.stable_baseline:
            self.use_softmax_attn = False  # it causes an error for imitation learning


        ### Human network ###
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(human_input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        if self.with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        mlp3_input_dim = mlp2_dims[-1] + self.robot_state_shape  
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None  # not used for now
        ###
        #if self.learn_no_human_embedding:
        #    self.no_human_embedding = nn.Parameter(torch.empty(self.human_out_dim), requires_grad=True)
        #    nn.init.xavier_uniform_(self.no_human_embedding.unsqueeze(0))  # while kaiming_uniform_ is default, its mainly designed for relu, so we use xavier_uniform_
        ###

        # try different initialization
        if self.learn_no_human_embedding:
            self.no_human_embedding = nn.Parameter(torch.empty(self.human_out_dim), requires_grad=True)
            nn.init.normal_(self.no_human_embedding, mean=0.0, std=0.1)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5) 
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # lidar
        if self.add_lidar:
            self.conv1 = nn.Conv1d(self.lidar_hist_steps, channel1, kernel1, stride=stride)  # 32, 5
            self.conv2 = nn.Conv1d(channel1, channel2, kernel2, stride=stride)  # 32, 3

            # Flatten (compute size of output)
            L_out = compute_1dconv_size(L_in=self.lidar_num_rays, kernel=kernel1, stride=stride)
            L_out = compute_1dconv_size(L_in=L_out, kernel=kernel2, stride=stride)
            self.lidar_linear = nn.Linear(channel2 * L_out, lidar_out)

        # linear
        self.linear1 = nn.Linear(self.robot_state_shape + lidar_out + self.human_out_dim, FC1)
        self.ln_linear1 = nn.LayerNorm(FC1)
        self.linear2 = nn.Linear(FC1, FC2)

        self.mean_linear = nn.Linear(FC2, actions_dim)
        self.value_linear = nn.Linear(FC2, 1)

        if self.fixed_sigma:
            self.sigma_linear = nn.Parameter(torch.zeros(actions_dim, requires_grad=True, dtype=torch.float32), requires_grad=True)
        else:
            self.sigma_linear = torch.nn.Linear(FC2, actions_dim)

    def is_separate_critic(self):
        return False

    def is_rnn(self):
        return False


    def forward(self, obs_dict):
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN found in parameter: {name}")
                print(f"Parameter shape: {param.shape}")
                print(f"NaN count: {torch.isnan(param).sum().item()}")
        
        obs = obs_dict
        # print(f"obs shape: {obs.shape}")
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if torch.isnan(obs).any():
            print(f"obs is nan: {obs}")
        batch_size = obs.size(0)
        robot_state = obs[:, 0:4]  # Robot [vx,vy,gx,gy]
        human_obs = obs[:, 4 + self.lidar_hist_steps * self.lidar_num_rays:]
        human_obs = human_obs.reshape(batch_size, self.humans_obs_limit, self.human_state_shape)

        if self.add_lidar:
            lidar = obs[:, 4: 4 + self.lidar_hist_steps * self.lidar_num_rays]
            lidar = lidar.reshape(batch_size, self.lidar_hist_steps, self.lidar_num_rays)
            lidar = F.leaky_relu(self.conv1(lidar), negative_slope=0.01)  # relu
            lidar = F.leaky_relu(self.conv2(lidar), negative_slope=0.01)  # relu
            lidar = torch.flatten(lidar, start_dim=1)
            lidar = self.lidar_linear(lidar)

        if self.with_om:  #
            human_output = self.get_human_output(robot_state=robot_state, human_obs=human_obs)
        else:
            human_output = self.get_human_output_vec(robot_state=robot_state, human_obs=human_obs)

        # cat + FC
        if self.add_lidar:
            x = torch.cat([robot_state, lidar, human_output], axis=-1)
        else:
            x = torch.cat([robot_state, human_output], axis=-1)
        
        if self.stable_baseline:
            return x, x
        
        # FC1-2
        x = self.linear1(x)
        x = F.leaky_relu(self.ln_linear1(x), negative_slope=0.01)
        x = F.leaky_relu(self.linear2(x), negative_slope=0.01)

        # action activation functions (to enforce bound)
        action = self.mean_linear(x)
        if self.kinematics == "holonomic":
            action = torch.tanh(action)  # so vel [-1, 1]
        elif self.kinematics == "diff_drive":
            action[:, 0] = F.sigmoid(action[:, 0])  # so vel [0, 1]
            action[:, 1] = torch.tanh(action[:, 1])  # so omega [-1, 1]

        value = self.value_linear(x)

        if self.fixed_sigma:
            sigma = self.sigma_linear
        else:
            sigma = self.sigma_linear(x)

        # debug
        #if torch.isnan(action).any():
        #    pdb.set_trace()

        if self.is_value_function:
            if value.shape[0] == 1:
                value = value.squeeze(0)
            return value
        else:
            if torch.isnan(action).any():
                print(f"action is nan: {action}")
            return action

    def forward_actor(self, obs):
        """
        Used for stable-baseline3
        """
        output, _ = self.forward(obs)
        return output

    def forward_critic(self, obs):
        """
        Used for stable-baseline3
        """
        output, _ = self.forward(obs)
        return output



    def get_human_output_vec(self, robot_state, human_obs):  # no occupancy_map
        """
        human_obs is [env, human_i, [px,py,vx,vy] ]
        robot_state is [env, [vx,vy, gx,gy] ]
        humans_state is [human_i, human_state cat robot_state (output of rotate func) cat occupancy map vector]

        rotate (func from SARL): takes in [robot_state(p,v,g), human_state(map_frame)]  ->  [robot_state(vx,vy towards goal, dg), human_state(robot_frame)]
        human_obs contains padding (humans with zero state that needs to be removed) which is removed here

        Runs per env since each env has uniqe number of humans (without padding) and padding won't work with attn
        For empty env, we return a zero tensor.
        1. get human_state+robot_state
        2. concat with occupancy map (optional)
        3. get state embedding (2 MLP layers)
        4. add mean embedding (global state) to all embeddings
        5. compute attention scores using attention MLP layer

            (1) This function can be converted into vectorized, by keeping padded humans and dealing with them by masking the attention scores
                scores = scores.masked_fill(~mask.squeeze(-1), float('-inf'))  # Assign -inf to padded humans
        """
        assert not self.with_om
        device = human_obs.device
        batch_size, max_num_humans, human_obs_dim = human_obs.shape

        # Create mask for real humans (non-padded)
        human_mask = torch.any(human_obs != 0, dim=2)  # [batch_size, max_num_humans]
        valid_humans_per_env = human_mask.sum(dim=1)  # [batch_size]

        # Expand robot_state to match human_obs for concatenation
        # [batch_size, 1, robot_state_dim] -> [batch_size, max_num_humans, robot_state_dim]
        robot_state_expanded = robot_state.unsqueeze(1).expand(-1, max_num_humans, -1)

        # Concatenate robot_state with human_obs
        # Assuming rotate function has already been applied externally or integrated here
        humans_state = torch.cat([robot_state_expanded, human_obs], dim=2)  # [batch_size, max_num_humans, robot_state_dim + human_obs_dim]
        # Pass through MLP layers
        mlp1_output = self.mlp1(humans_state)  # [batch_size, max_num_humans, mlp1_dim]
        mlp2_output = self.mlp2(mlp1_output)  # [batch_size, max_num_humans, mlp2_dim]

        if self.with_global_state:
            # Compute global state as mean (after removing padded humans) over humans
            human_mask_expanded = human_mask.unsqueeze(-1)  # [batch_size, max_num_humans, 1]
            masked_mlp1_output = mlp1_output * human_mask_expanded  # [batch_size, max_num_humans, mlp1_dim]
            # Compute sum over valid humans
            sum_valid_humans = masked_mlp1_output.sum(dim=1, keepdim=True)  # [batch_size, 1, mlp1_dim]
            # Count number of valid humans per environment, avoiding division by zero
            valid_humans_count = human_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)  # [batch_size, 1, 1]
            # mean
            global_state_mean = sum_valid_humans / valid_humans_count  # [batch_size, 1, mlp1_dim]
            global_state_expanded = global_state_mean.expand(-1, max_num_humans, -1)  # [batch_size, max_num_humans, mlp1_dim]
            attention_input = torch.cat([mlp1_output, global_state_expanded], dim=2)  # [batch_size, max_num_humans, mlp1_dim + mlp1_dim]
        else:
            attention_input = mlp1_output  # [batch_size, max_num_humans, mlp1_dim]

        # Compute attention scores
        scores = self.attention(attention_input).squeeze(-1)  # [batch_size, max_num_humans]

        # Mask padded humans by setting their scores to -inf
        scores = scores.masked_fill(~human_mask, float('-inf'))  # [batch_size, max_num_humans]

        if self.use_softmax_attn:
            # Apply masked softmax; ensure that environments with no humans have weights zero
            weights = F.softmax(scores, dim=1)  # [batch_size, max_num_humans]
        else:
            # Apply masked exponentiation and normalization
            scores_exp = torch.exp(scores) * human_mask.float()  # [batch_size, max_num_humans]
            weights = scores_exp / (scores_exp.sum(dim=1, keepdim=True) + 1e-8)  # [batch_size, max_num_humans]

        # Compute weighted features
        # [batch_size, max_num_humans, mlp2_dim] * [batch_size, max_num_humans, 1] -> [batch_size, max_num_humans, mlp2_dim]
        weighted_features = mlp2_output * weights.unsqueeze(-1)  # [batch_size, max_num_humans, mlp2_dim]
        weighted_feature = weighted_features.sum(dim=1)  # [batch_size, mlp2_dim]

        joint_state = torch.cat([robot_state, weighted_feature], dim=1)  # [batch_size, robot_state_dim + mlp2_dim]
        human_output = self.mlp3(joint_state)  # [batch_size, human_out_dim]

        # Handle environments with no humans
        if self.learn_no_human_embedding:
            no_human_embedding_expanded = self.no_human_embedding.unsqueeze(0).expand(batch_size, -1).to(device)  # [batch_size, no_human_dim]
            # Create a mask for environments with no humans
            no_humans = (valid_humans_per_env == 0).unsqueeze(-1)  # [batch_size, 1]
            # Replace weighted_feature with no_human_embedding where there are no humans
            human_output = torch.where(no_humans, no_human_embedding_expanded, human_output)  # [batch_size, mlp2_dim]
        else:
            # For environments with no humans, weighted_feature remains zero
            no_humans = (valid_humans_per_env == 0)  # [batch_size, 1]
            human_output[no_humans] = 0.


        return human_output




    def get_human_output(self, robot_state, human_obs): 
        """
        human_obs is [env, human_i, [px,py,vx,vy] ]
        robot_state is [env, [vx,vy, gx,gy] ]
        humans_state is [human_i, human_state cat robot_state (output of rotate func) cat occupancy map vector]

        rotate (func from SARL): takes in [robot_state(p,v,g), human_state(map_frame)]  ->  [robot_state(vx,vy towards goal, dg), human_state(robot_frame)]
        human_obs contains padding (humans with zero state that needs to be removed) which is removed here

        Runs per env since each env has uniqe number of humans (without padding) and padding won't work with attn
        For empty env, we return a zero tensor.
        1. get human_state+robot_state
        2. concat with occupancy map (optional)
        3. get state embedding (2 MLP layers)
        4. add mean embedding (global state) to all embeddings
        5. compute attention scores using attention MLP layer
        """
        device = human_obs.device
        batch_size = human_obs.shape[0]

        human_output = torch.zeros((batch_size, self.human_out_dim), device=device, dtype=human_obs.dtype)
        if self.learn_no_human_embedding:
            no_human_embedding_expanded = self.no_human_embedding.expand(batch_size, -1)

        for env_idx in range(batch_size):
            # robot_state before each human # dim is [env, human_i, 8]
            batch_robot_state = robot_state[env_idx]
            batch_human_obs = human_obs[env_idx] 
            batch_human_obs = batch_human_obs[torch.any(batch_human_obs != 0, dim=1)]  # remove padding

            if batch_human_obs.size(0) == 0 and self.learn_no_human_embedding:  # case of no humans
                human_output[env_idx] = no_human_embedding_expanded[env_idx]  # used learnable parameter
            if batch_human_obs.size(0) == 0:  # case of no humans
                continue

            humans_state = torch.cat([batch_robot_state.unsqueeze(0).expand(batch_human_obs.size(0), -1), batch_human_obs], dim=1)  # cat each batch_i
            
            if self.with_om:  
                # dim is [human_i, cell_num^2 * channel]
                occupancy_maps = self.build_occupancy_maps(batch_human_obs)
                humans_state = torch.cat([humans_state, occupancy_maps], dim=1)

            size = humans_state.shape  # size[0]: env, size[1]: human, size[2]: human_state(4+4+ cell_num^2*channel)
            mlp1_output = self.mlp1(humans_state)  
            mlp2_output = self.mlp2(mlp1_output)

            if self.with_global_state:
                # compute attention scores
                global_state = torch.mean(mlp1_output, dim=0, keepdim=True)
                # [1,global_dim] -> [human_i, global_dim] # contiguous: saves expand in memory,  
                global_state = global_state.expand((size[0], self.global_state_dim)).contiguous()#.view(-1, self.global_state_dim)
                attention_input = torch.cat([mlp1_output, global_state], dim=1)
            else:
                attention_input = mlp1_output
            scores = self.attention(attention_input).view(size[0], 1).squeeze(dim=1)  # [human_i] single score per human 

            # masked softmax

            if self.use_softmax_attn:
                weights = F.softmax(scores, dim=0).unsqueeze(-1)
            else:
                scores_exp = torch.exp(scores) #* (scores != 0).float()  # part of original but doesn't seem to be needed
                weights = (scores_exp / torch.sum(scores_exp, dim=0, keepdim=True)).unsqueeze(1)  # [human_i, 1]

            #self.attention_weights = weights[0, :, 0].data.cpu().numpy()

            # output feature is a linear combination of input features
            features = mlp2_output  # [human_i, mlp2_dim[-1]]
            weighted_feature = torch.sum(torch.mul(weights, features), dim=0)  # mul: each fature times weight # sum: add all features to get single feature

            # concatenate agent's state with global weighted humans' state
            joint_state = torch.cat([batch_robot_state, weighted_feature], dim=0)
            human_output[env_idx] = self.mlp3(joint_state)

        return human_output 






    def build_occupancy_maps(self, human_states):  
        """
        # TODO: would it work if the humans are robot-centric, since orig paper uses map coordinates

        :param human_states: [human_i, [px,py,vx,vy]] (no padding or zero-humans)
        :return: tensor of shape (# human - 1, self.cell_num ** 2)
        """
        a, b = human_states.shape
        assert b == 4
        device = human_states.device
        
        occupancy_maps = torch.zeros((human_states.shape[0], (self.cell_num**2) * self.om_channel_size), device=device, dtype=human_states.dtype)
        for idx, human in enumerate(human_states):
            other_humans = human_states[torch.arange(human_states.size(0), device=device) != idx]
            other_px = other_humans[:, 0] - human[0]
            other_py = other_humans[:, 1] - human[1]
            # new x-axis is in the direction of human's velocity
            human_velocity_angle = torch.arctan2(human[3], human[2])
            other_human_orientation = torch.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = torch.norm(torch.stack([other_px, other_py], dim=0), dim=0)
            other_px = torch.cos(rotation) * distance
            other_py = torch.sin(rotation) * distance

            # compute indices of humans in the grid
            other_x_index = torch.floor(other_px / self.cell_size + self.cell_num / 2).long()
            other_y_index = torch.floor(other_py / self.cell_size + self.cell_num / 2).long()
            other_x_index[other_x_index < 0] = int(-1e12)
            other_x_index[other_x_index >= self.cell_num] = int(-1e12)
            other_y_index[other_y_index < 0] = int(-1e12)
            other_y_index[other_y_index >= self.cell_num] = int(-1e12)
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = torch.isin(torch.arange(self.cell_num ** 2, device=device), grid_indices)
            if self.om_channel_size == 1:
                occupancy_maps[idx] = occupancy_map.to(torch.long)
            else:
                # calculate relative velocity for other agents
                other_human_velocity_angles = torch.arctan2(other_humans[:, 3], other_humans[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = torch.norm(other_humans[:, 2:4], dim=1)
                other_vx = torch.cos(rotation) * speed
                other_vy = torch.sin(rotation) * speed

                ## remove indcies that are negative (outside the map)
                valid_mask = grid_indices >= 0
                grid_indices = grid_indices[valid_mask]
                other_vx = other_vx[valid_mask]
                other_vy = other_vy[valid_mask]

                dm_sums = torch.zeros(self.om_dim, dtype=human_states.dtype, device=device)
                dm_counts = torch.zeros(self.om_dim, dtype=human_states.dtype, device=device)
                if self.om_channel_size == 2:
                    vx_indices = 2 * grid_indices
                    vy_indices = 2 * grid_indices + 1

                    # Update sums and counts
                    dm_sums.index_add_(0, vx_indices, other_vx)
                    dm_counts.index_add_(0, vx_indices, torch.ones_like(other_vx, device=device))
                    dm_sums.index_add_(0, vy_indices, other_vy)
                    dm_counts.index_add_(0, vy_indices, torch.ones_like(other_vy, device=device))

                elif self.om_channel_size == 3:
                    const_indices = 3 * grid_indices
                    vx_indices = 3 * grid_indices + 1
                    vy_indices = 3 * grid_indices + 2

                    # Update sums and counts
                    dm_sums.index_add_(0, const_indices, torch.ones_like(grid_indices, dtype=human_states.dtype, device=device))
                    dm_counts.index_add_(0, const_indices, torch.ones_like(grid_indices, dtype=human_states.dtype, device=device))
                    dm_sums.index_add_(0, vx_indices, other_vx)
                    dm_counts.index_add_(0, vx_indices, torch.ones_like(other_vx, device=device))
                    dm_sums.index_add_(0, vy_indices, other_vy)
                    dm_counts.index_add_(0, vy_indices, torch.ones_like(other_vy, device=device))

                else:
                    raise NotImplementedError

                dm_torch = torch.where(dm_counts > 0, dm_sums / dm_counts, torch.zeros_like(dm_sums, device=device))

                #assert torch.allclose(dm_torch, dm, atol=1e-6), pdb.set_trace()
                occupancy_maps[idx] = dm_torch
        return occupancy_maps



def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net
















  
