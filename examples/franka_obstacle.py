import copy
from isaacgym import gymapi
from isaacgym import gymutil

import torch

torch.multiprocessing.set_start_method('spawn', force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


import time
import yaml
import argparse
import numpy as np
from quaternion import quaternion, from_rotation_vector, from_rotation_matrix

from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array

from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict

from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path

from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.mpc.task.reacher_task import ReacherTask
from torch.distributions.gamma import Gamma
from scipy import stats
import datetime


np.set_printoptions(precision=2)

debug = False


def calculateLognormalParams(mean, std):
    location = np.log(mean ** 2 / np.sqrt(std ** 2 + mean ** 2))
    scale = np.sqrt(np.log((std/mean) ** 2 + 1))
    return (location, scale)

def calculateGammaParams(mean, std):
    alpha = (mean/std)**2
    beta = mean/std**2
    return (alpha, beta)


def sample_gamma(mean, std):
    alpha, beta = calculateGammaParams(mean.item(), std)
    sample = stats.gamma.rvs(a=alpha, loc=0, scale=1/beta)
    return sample

cuda = True

class FrankaSimulation():

    def __init__(self, gym_instance, target_position, world_file=None):

        self.gym_instance = gym_instance
        self.vis_ee_target = True
        self.robot_file = 'franka.yml'
        self.task_file = 'franka_reacher.yml'
        self.world_file = world_file if world_file else 'collision_primitives_3d.yml'

        self.gym = self.gym_instance.gym
        sim = self.gym_instance.sim
        world_yml = join_path(get_gym_configs_path(), self.world_file)
        with open(world_yml) as file:
            world_params = yaml.load(file, Loader=yaml.FullLoader)

        robot_yml = join_path(get_gym_configs_path(), 'franka.yml')
        with open(robot_yml) as file:
            robot_params = yaml.load(file, Loader=yaml.FullLoader)
        sim_params = robot_params['sim_params']
        sim_params['asset_root'] = get_assets_path()
        if (cuda):
            device_name = 'cuda'
        else:
            device_name = 'cpu'
        print('+++++++++++++++++++++++++++', device_name)

        sim_params['collision_model'] = None
        # create robot simulation:
        self.robot_sim = RobotSim(gym_instance=self.gym, sim_instance=sim, **sim_params, device=device_name)

        # create gym environment:
        robot_pose = sim_params['robot_pose']
        self.env_ptr = self.gym_instance.env_list[0]
        self.robot_ptr = self.robot_sim.spawn_robot(self.env_ptr, robot_pose, coll_id=2)

        device = torch.device(device_name, 0)

        self.tensor_args = {'device': device, 'dtype': torch.float32}

        # spawn camera:
        robot_camera_pose = np.array([1.6, -1.5, 1.8, 0.707, 0.0, 0.0, 0.707])
        q = as_float_array(from_euler_angles(-0.5 * 90.0 * 0.01745, 50.0 * 0.01745, 90 * 0.01745))
        robot_camera_pose[3:] = np.array([q[1], q[2], q[3], q[0]])

        self.robot_sim.spawn_camera(self.env_ptr, 60, 640, 480, robot_camera_pose)

        # get pose
        self.w_T_r = copy.deepcopy(self.robot_sim.spawn_robot_pose)

        w_T_robot = torch.eye(4)
        quat = torch.tensor([self.w_T_r.r.w, self.w_T_r.r.x, self.w_T_r.r.y, self.w_T_r.r.z]).unsqueeze(0)
        rot = quaternion_to_matrix(quat)
        w_T_robot[0, 3] = self.w_T_r.p.x
        w_T_robot[1, 3] = self.w_T_r.p.y
        w_T_robot[2, 3] = self.w_T_r.p.z
        w_T_robot[:3, :3] = rot[0]

        self.world_instance = World(self.gym, sim, self.env_ptr, world_params, w_T_r=self.w_T_r)

        self.mpc_control = ReacherTask(self.task_file, self.robot_file, self.world_file, self.tensor_args)

        mpc_tensor_dtype = {'device': device, 'dtype': torch.float32}


        # spawn object:
        x, y, z = 0.0, 0.0, 0.0
        tray_color = gymapi.Vec3(0.8, 0.7, 0.1)
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.thickness = 0.002

        object_pose = gymapi.Transform()
        object_pose.p = gymapi.Vec3(x, y, z)
        object_pose.r = gymapi.Quat(0, 0, 0, 1)

        obj_asset_file = "urdf/mug/movable_mug.urdf"
        obj_asset_root = get_assets_path()

        if self.vis_ee_target:
            target_object = self.world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, color=tray_color,
                                                        name='ee_target_object')
            obj_base_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, target_object, 0)
            self.obj_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, target_object, 6)
            self.gym.set_rigid_body_color(self.env_ptr, target_object, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
            self.gym.set_rigid_body_color(self.env_ptr, target_object, 6, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)

            obj_asset_file = "urdf/mug/mug.urdf"
            obj_asset_root = get_assets_path()

            ee_handle = self.world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, color=tray_color,
                                                    name='ee_current_as_mug')
            self.ee_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, ee_handle, 0)
            tray_color = gymapi.Vec3(0.0, 0.8, 0.0)
            self.gym.set_rigid_body_color(self.env_ptr, ee_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)

        ## TARGET POSITION
        self.g_pos = target_position
        # self.g_pos = np.array([0.2, -0.2, 0.05])
        # self.g_pos = np.array([0.5, -0.2, 0.05])
        self.g_q = np.array([0.04, 0.43, 0.9, 0.05]) # DEFAULT FROM THE EXAMPLE

        object_pose.p = gymapi.Vec3(self.g_pos[0], self.g_pos[1], self.g_pos[2])
        object_pose.r = gymapi.Quat(self.g_q[1], self.g_q[2], self.g_q[3], self.g_q[0])
        object_pose = self.w_T_r * object_pose
        if (self.vis_ee_target):
            self.gym.set_rigid_transform(self.env_ptr, obj_base_handle, object_pose)
        self.ee_pose = gymapi.Transform()
        self.w_robot_coord = CoordinateTransform(trans=w_T_robot[0:3, 3].unsqueeze(0),
                                            rot=w_T_robot[0:3, 0:3].unsqueeze(0))

        self.tensor_args = mpc_tensor_dtype


    def step_sampling(self, d1, d2, d3):
        ## Change world collision per roll-out
        self.mpc_control.change_collision_params(sample_gamma, d1, d2, d3)


    def change_dimensions(self, d_search=None):
        dimx = d_search['d1']
        dimy = d_search['d2']
        dimz = None

        world_collision = {'coll_objs': {'cube': {'obstacle1': {'dims': [0.3, 0.1, 0.6], 'pose': [0.4, 0.08, 0.2, 0, 0, 0, 1.0]},
                                                  'table': {'dims': [2.0, 2.0, 0.2], 'pose': [0.0, 0.0, -0.1, 0, 0, 0, 1.0]}}}}
        if dimx != None:
            world_collision['coll_objs']['cube']['obstacle1']['dims'][0] = dimx
        if dimy != None:
            world_collision['coll_objs']['cube']['obstacle1']['dims'][1] = dimy
        if dimz != None:
            world_collision['coll_objs']['cube']['obstacle1']['dims'][2] = dimz
        print('dims:', world_collision['coll_objs']['cube']['obstacle1']['dims'])


    def run_episode(self, target_position, steps, distributed_dimensions, **varying_params):

        self.mpc_control.initialize_mpc()
        if 'beta' in varying_params and 'init_cov' in varying_params:
            self.mpc_control.change_params(varying_params)

        ################## Goal ROBOT state??
        franka_bl_state = np.array([-0.3, 0.3, 0.2, -2.0, 0.0, 2.4, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        x_des_list = [franka_bl_state]
        i = 0
        x_des = x_des_list[0]
        self.mpc_control.update_params(goal_state=x_des)
        self.sim_dt = self.mpc_control.exp_params['control_dt']
        ##################

        CONTROL = True
        SHOW_TRAJECTORIES = False
        GRIPPER_THING = True

        # Cumulative Episodic Cost
        cum_cost = 0
        step = 1

        self.t_step = 0

        while step <= steps:
            try:
                if debug: print('Step:', step)
                self.gym_instance.step()

                self.t_step += self.sim_dt

                if CONTROL:

                    if distributed_dimensions:
                        self.step_sampling(varying_params['d1'], varying_params['d2'], varying_params['d3'])

                    current_robot_state = copy.deepcopy(self.robot_sim.get_state(self.env_ptr, self.robot_ptr))
                    action_cost, command = self.mpc_control.get_command(self.t_step, current_robot_state, control_dt=self.sim_dt, WAIT=True)

                    filtered_state_mpc = current_robot_state  # mpc_control.current_state
                    curr_state = np.hstack(
                        (filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))

                    ## gripper thing
                    if GRIPPER_THING:
                        curr_state = np.hstack(
                            (filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))
                        curr_state_tensor = torch.as_tensor(curr_state, **self.tensor_args).unsqueeze(0)
                        pose_state = self.mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
                        e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
                        e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
                        self.ee_pose.p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2]))
                        self.ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])
                        self.ee_pose = copy.deepcopy(self.w_T_r) * copy.deepcopy(self.ee_pose)

                        if (self.vis_ee_target):
                            self.gym.set_rigid_transform(self.env_ptr, self.ee_body_handle, copy.deepcopy(self.ee_pose))

                    if SHOW_TRAJECTORIES:
                        self.gym_instance.clear_lines()
                        top_trajs = self.mpc_control.top_trajs.cpu().float()
                        n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
                        w_pts = self.w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)

                        top_trajs = w_pts.cpu().numpy()
                        color = np.array([0.0, 1.0, 0.0])
                        # red trajectory = last element of top_trajs
                        # no special order, top 10 trajectories of 150 samples (particles)
                        for k in range(top_trajs.shape[0]):
                            pts = top_trajs[k, :, :]
                            color[0] = float(k) / float(top_trajs.shape[0])
                            color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
                            self.gym_instance.draw_lines(pts, color=color)

                    q_des = copy.deepcopy(command['position'])
                    self.robot_sim.command_robot_position(q_des, self.env_ptr, self.robot_ptr)

                    if debug: print('COST:', action_cost)
                    cum_cost += action_cost

                if CONTROL:
                    step += 1

            except KeyboardInterrupt:
                # print('Closing')
                done = True
                break

        return cum_cost.item() if torch.is_tensor(cum_cost) else cum_cost


def run_experiment(episodes, target_position, initial_robot_state, franka_simulation, gym_instance, sim_params, x, return_cumcosts=False):
    x = np.array(x).flatten()
    varying_params = {
        "beta": x[0],
        "init_cov": x[1]
    }

    steps = 500
    cum_costs = []

    for episode in range(episodes):
        print('###### EPISODE:', episode + 1)
        franka_simulation.robot_sim.set_robot_state(initial_robot_state['position'], initial_robot_state['velocity'], franka_simulation.env_ptr, franka_simulation.robot_ptr)
        cum_cost = franka_simulation.run_episode(target_position=target_position, steps=steps, distributed_dimensions=False, **varying_params)
        print('CUMULATIVE COST:', cum_cost)
        cum_costs.append(cum_cost)

    if return_cumcosts:
        return cum_costs

    mean_cost = np.nanmean(cum_costs)
    print('MEAN EPISODIC COST:', mean_cost)
    return mean_cost



if __name__ == '__main__':

    sim_params = load_yaml(join_path(get_gym_configs_path(), 'physx.yml'))
    sim_params['headless'] = True

    gym_instance = Gym(**sim_params)

    target_position = np.array([ 0.62, -0.06,  0.22])
    franka_simulation = FrankaSimulation(gym_instance, target_position=target_position)
    initial_robot_state = copy.deepcopy(franka_simulation.robot_sim.get_state(franka_simulation.env_ptr, franka_simulation.robot_ptr))

    episodes = 50
    mean_cost = run_experiment(episodes, target_position, initial_robot_state, franka_simulation, gym_instance, sim_params, x=[3, 0.5])
    print(mean_cost)
