#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
import torch
import yaml
import numpy as np

from ...util_file import get_mpc_configs_path as mpc_configs_path
from ...mpc.rollout.arm_reacher import ArmBase
from ...mpc.control import MPPI
from ...mpc.utils.state_filter import JointStateFilter
from ...mpc.utils.mpc_process_wrapper import ControlProcess
from ...util_file import get_assets_path, join_path, load_yaml, get_gym_configs_path
from .task_base import BaseTask

from ..cost import CollisionCost, VoxelCollisionCost, PrimitiveCollisionCost



class ArmTask(BaseTask):
    def __init__(self, task_file='ur10.yml', robot_file='ur10_reacher.yml', world_file='collision_env.yml', tensor_args={'device':"cpu", 'dtype':torch.float32}):

        super().__init__(tensor_args=tensor_args)
        self.init_mppi(task_file, robot_file, world_file)

    def print_params(self):
        print("[{}, {}, {}, {}, {}, {}, {}]".format(self.controller.beta, self.controller.init_cov,
                                                    self.controller.gamma, self.controller.alpha, self.controller.kappa,
                                                    self.controller.step_size_mean, self.controller.step_size_cov))

    def change_params(self, varying_params):
        if 'beta' in varying_params and varying_params['beta'] != None:
            # print('beta update')
            self.controller.beta = varying_params['beta']
        # if 'gamma' in varying_params and varying_params['gamma'] != None:
        #     self.controller.gamma = varying_params['gamma']
        if 'alpha' in varying_params and varying_params['alpha'] != None:
            self.controller.alpha = varying_params['alpha']
        if 'kappa' in varying_params and varying_params['kappa'] != None:
            self.controller.kappa = varying_params['kappa']
        if 'step_size_mean' in varying_params and varying_params['step_size_mean'] != None:
            self.controller.step_size_mean = varying_params['step_size_mean']
        if 'step_size_cov' in varying_params and varying_params['step_size_cov'] != None:
            self.controller.step_size_cov = varying_params['step_size_cov']

    def change_collision_params(self, sampling_distr, d1, d2, d3):

        self.rollout_fn.primitive_collision_cost.robot_world_coll.world_coll.change_collision_cube(sampling_distr, d1, d2, d3)

    def get_rollout_fn(self, **kwargs):
        rollout_fn = ArmBase(**kwargs)
        return rollout_fn

    def init_mppi(self, task_file, robot_file, collision_file):
        robot_yml = join_path(get_gym_configs_path(), robot_file)

        with open(robot_yml) as file:
            robot_params = yaml.load(file, Loader=yaml.FullLoader)

        world_yml = join_path(get_gym_configs_path(), collision_file)
        with open(world_yml) as file:
            world_params = yaml.load(file, Loader=yaml.FullLoader)

        mpc_yml_file = join_path(mpc_configs_path(), task_file)

        with open(mpc_yml_file) as file:
            self.exp_params = yaml.load(file, Loader=yaml.FullLoader)
        self.exp_params['robot_params'] = self.exp_params['model'] #robot_params


        self.rollout_fn = self.get_rollout_fn(exp_params=self.exp_params, tensor_args=self.tensor_args, world_params=world_params)

        self.mppi_params = self.exp_params['mppi']
        dynamics_model = self.rollout_fn.dynamics_model
        self.mppi_params['d_action'] = dynamics_model.d_action
        self.mppi_params['action_lows'] = -self.exp_params['model']['max_acc'] * torch.ones(dynamics_model.d_action, **self.tensor_args)
        self.mppi_params['action_highs'] = self.exp_params['model']['max_acc'] * torch.ones(dynamics_model.d_action, **self.tensor_args)
        init_q = torch.tensor(self.exp_params['model']['init_state'], **self.tensor_args)
        init_action = torch.zeros((self.mppi_params['horizon'], dynamics_model.d_action), **self.tensor_args)
        init_action[:,:] += init_q
        if(self.exp_params['control_space'] == 'acc'):
            self.mppi_params['init_mean'] = init_action * 0.0 # device=device)
        elif(self.exp_params['control_space'] == 'pos'):
            self.mppi_params['init_mean'] = init_action
        self.mppi_params['rollout_fn'] = self.rollout_fn
        self.mppi_params['tensor_args'] = self.tensor_args

    def initialize_mpc(self):
        self.controller = MPPI(**self.mppi_params)
        self.initialize_control_process()
        self.init_aux()
        # self.exp_params = exp_params
        # return controller

