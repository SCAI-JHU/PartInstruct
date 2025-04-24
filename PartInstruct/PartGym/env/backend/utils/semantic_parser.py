from pathlib import Path
import os
from typing import Union, Optional
from collections import defaultdict

import numpy as np
import pybullet

from PartInstruct.PartGym.env.backend.bullet_sim import *
from PartInstruct.PartGym.env.backend.utils.vision_utils import *

import json
from copy import deepcopy
from numpy.typing import ArrayLike

class SemanticParser:
    def __init__(self, world: BtWorld):
        ## Setups
        self.spatial_sampler = SpatialSampler()
        self.world = world
        self.obj = None
        ## Info
        self.obj_name = None
        self.part_hierarchy = None
        self.part_pcds_t0 = {}
        self.part_pcds_last_time = {}
        self.transform_last_time = Transform(Rotation.identity(), [0, 0, 0])
        self.part_pcds = {}

    def load_part_hierarchy(self, part_path, scale=1.0, num_points=3000):
        ## TODO keep a constant pcd sampling density
        
        result_path = os.path.join(part_path, "result_after_merging.json")
        if not os.path.exists(result_path):
            result_path = os.path.join(part_path, "result.json")

        with open(result_path) as f:
            self.part_hierarchy = json.load(f)
        
        self.obj_name = self.part_hierarchy[0]["name"].lower()

        if "result.json" in result_path:
            combined_point_cloud = self.load_combined_point_cloud_from_objs(part_path, num_points)
            self.part_pcds_t0[self.obj_name] = combined_point_cloud * scale
        else:
            ply_path = os.path.join(part_path, "point_sample", "ply-10000.ply")
            pcd_obj = np.asarray(o3d.io.read_point_cloud(ply_path).points)
            self.part_pcds_t0[self.obj_name] = pcd_obj * scale

        part_point_clouds = defaultdict(lambda: np.array([]).reshape(0, 3))

        def process_node(node):
            if "children" not in node or len(node["children"]) == 0:  # Leaf node
                part_name = node["name"].lower()
                if 'other' in part_name:
                    return
                if '_' in part_name:
                    part_name = ' '.join(part_name.split('_'))

                combined_point_cloud = np.array([]).reshape(0, 3)  # Initialize an empty array for the combined point cloud
                
                # Combine point clouds from all .obj files listed for this part
                for obj_file in node["objs"]:
                    obj_path = os.path.join(part_path, "objs", obj_file + ".obj")
                    if not os.path.exists(obj_path):
                        obj_path = os.path.join(part_path, "textured_objs", obj_file + ".obj")
                    point_cloud = sample_point_cloud_from_obj(obj_path, num_points)
                    
                    # Combine the current point cloud with the previously accumulated ones
                    combined_point_cloud = np.vstack((combined_point_cloud, point_cloud)) if combined_point_cloud.size else point_cloud

                # Combine the point cloud for the current part with any previously accumulated point clouds for parts with the same name
                part_point_clouds[part_name] = np.vstack((part_point_clouds[part_name], combined_point_cloud)) if part_point_clouds[part_name].size else combined_point_cloud
            else:
                # Recursively process children nodes
                for child in node["children"]:
                    process_node(child)
        
        process_node(self.part_hierarchy[0])

        # Save the combined point clouds to self.part_pcds_t0
        for part_name, combined_point_cloud in part_point_clouds.items():
            self.part_pcds_t0[part_name] = combined_point_cloud * scale

        for locative_noun in self.spatial_sampler.locative_nouns:
            self.part_pcds_t0[locative_noun] = self.spatial_sampler.sample_query(self.part_pcds_t0[self.obj_name], locative_noun)

        self.part_pcds_last_time = deepcopy(self.part_pcds_t0)

        if self.obj:
            self.update_part_pcds(resample_spatial=True)
        else:
            self.update_part_pcds_by_transform(Rotation.identity(), [0, 0, 0], resample_spatial=True)

    def load_combined_point_cloud_from_objs(self, part_path, num_points, target_num_points=10000):
        objs_dir = os.path.join(part_path, "textured_objs")
        combined_point_cloud = np.array([]).reshape(0, 3)
        for obj_file in os.listdir(objs_dir):
            if obj_file.endswith(".obj"):
                obj_path = os.path.join(objs_dir, obj_file)
                point_cloud = sample_point_cloud_from_obj(obj_path, num_points)
                combined_point_cloud = np.vstack((combined_point_cloud, point_cloud)) if combined_point_cloud.size else point_cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_point_cloud)
        if len(pcd.points) > target_num_points:
            indices = np.random.choice(len(pcd.points), target_num_points, replace=False)
            downsampled_pcd = pcd.select_by_index(indices)
        else:
            downsampled_pcd = pcd
        downsampled_np_array = np.asarray(downsampled_pcd.points)
        return downsampled_np_array

    def set_obj(self, obj: Body):
        self.obj = obj

    def get_one_part_pcd(self, part: str):
        assert self.obj and self.part_pcds_t0 and part in self.part_pcds_t0
        position, orientation = self.world.get_body_pose(self.obj)
        current_body_pose = Transform(Rotation.from_quat(orientation), np.array(position))
        is_semantic = True if part not in self.spatial_sampler.locative_nouns else False

        if is_semantic:
            pcd = self.part_pcds_t0[part] # T_w_obj*p
            updated_pcd = transform_point_cloud(pcd, list(position), list(orientation))
        else:
            assert self.part_pcds_last_time
            # update part pcd from last time
            pcd = self.part_pcds_last_time[part]
            # calculate the relative transform from last time to this time
            # p_obj = T_obj_last * p_last
            relative_transform = current_body_pose*self.transform_last_time.inverse()
            updated_pcd = transform_point_cloud(pcd, list(relative_transform.translation), list(relative_transform.rotation.as_quat()))
        
        # p_cam = T_cam_w *p_w
        return updated_pcd

    def update_part_pcds(self, resample_spatial=False):
        assert self.obj and self.part_pcds_t0
        position, orientation = self.world.get_body_pose(self.obj)
        current_body_pose = Transform(Rotation.from_quat(orientation), np.array(position))
        part_semantic = [part for part in self.part_pcds_t0.keys() if part not in self.spatial_sampler.locative_nouns]
        part_spatial = list(self.spatial_sampler.locative_nouns)

        for part in part_semantic:
            pcd = self.part_pcds_t0[part] # T_w_cur = T_w_t0 * T_t0_cur; T_t0_cur = T_w_t0.inverse() * T_w_cur
            updated_pcd = transform_point_cloud(pcd, list(position), list(orientation))
            self.part_pcds[part] = updated_pcd
        # T_w_obj*x_obj_p
        # T_cam_w*x_w_p
        if not resample_spatial:
            assert self.part_pcds_last_time
            # update part pcd from last time
            for part in part_spatial:
                pcd = self.part_pcds_last_time[part]
                # calculate the relative transform from last time to this time
                relative_transform = current_body_pose*self.transform_last_time.inverse()
                updated_pcd = transform_point_cloud(pcd, list(relative_transform.translation), list(relative_transform.rotation.as_quat()))
                self.part_pcds[part] = updated_pcd
        else:
            for part in part_spatial:
                updated_pcd = self.spatial_sampler.sample_query(self.part_pcds[self.obj_name], part)
                self.part_pcds[part] = updated_pcd
            # update the last time part pcds
            self.part_pcds_last_time = deepcopy(self.part_pcds)
            position, orientation = self.world.get_body_pose(self.obj)
            self.transform_last_time = Transform(Rotation.from_quat(orientation), np.array(position))
            
        return deepcopy(self.part_pcds)

    def update_part_pcds_by_transform(self, rotation: Rotation, translation: ArrayLike=[0,0,0], resample_spatial=False):
        assert self.part_pcds_t0
        current_body_pose = Transform(rotation, translation)
        part_semantic = [part for part in self.part_pcds_t0.keys() if part not in self.spatial_sampler.locative_nouns]
        part_spatial = list(self.spatial_sampler.locative_nouns)
        orientation = rotation.as_quat()

        for part in part_semantic:
            pcd = self.part_pcds_t0[part]
            updated_pcd = transform_point_cloud(pcd, list(translation), list(orientation))
            self.part_pcds[part] = updated_pcd

        if not resample_spatial:
            assert self.part_pcds_last_time
            for part in part_spatial:
                pcd = self.part_pcds_last_time[part]
                relative_transform = current_body_pose*self.transform_last_time.inverse()
                updated_pcd = transform_point_cloud(pcd, list(relative_transform.translation), list(relative_transform.rotation.as_quat()))
                self.part_pcds[part] = updated_pcd
        else:
            for part in part_spatial:
                updated_pcd = self.spatial_sampler.sample_query(self.part_pcds[self.obj_name], part)
                self.part_pcds[part] = updated_pcd
            # update the last time part pcds
            self.part_pcds_last_time = deepcopy(self.part_pcds)
            self.transform_last_time = Transform(Rotation.from_quat(orientation), np.array(translation))

        return deepcopy(self.part_pcds)

class SpatialSampler:

    locative_nouns = ["front", "back", "top", "bottom", "left", "right", "middle"]

    gaussian_param_mappings = {
        'left': {'mu': [0, 1.0, 0], 'sigma': [0.2, 0.2, 0.2]},
        'right': {'mu': [0, -1.0, 0], 'sigma': [0.2, 0.2, 0.2]},
        'top': {'mu': [0, 0, 1.0], 'sigma': [0.2, 0.2, 0.2]},
        'bottom': {'mu': [0, 0, -1.0], 'sigma': [0.2, 0.2, 0.2]},
        'front': {'mu': [-1.0, 0, 0], 'sigma': [0.2, 0.2, 0.2]},
        'back': {'mu': [1.0, 0, 0], 'sigma': [0.2, 0.2, 0.2]},
        'middle': {'mu': [0, 0, 0], 'sigma': [0.03, 0.03, 0.03]},
    }

    def __init__(self):
        self.locative_nouns = SpatialSampler.locative_nouns

        self.gaussian_param_mappings = SpatialSampler.gaussian_param_mappings

        self.reference_rotation = np.eye(3)

    def set_reference_rotation(self, rotation):
        """ Set the reference rotation to align input point clouds to the desired coordinate system. """
        self.reference_rotation = rotation

    def apply_transformation(self, pcd):
        """ Apply the transformation matrix to the point cloud. """
        return np.dot(pcd, self.reference_rotation)

    def normalize_pcd(self, pcd):
        """ Normalize point cloud to have zero mean and unit scale. """
        min_point = np.min(pcd, axis=0)
        max_point = np.max(pcd, axis=0)
        center = (min_point + max_point) / 2
        pcd_centered = pcd - center
        pcd_centered = self.apply_transformation(pcd_centered)
        scale_factor = np.sqrt(np.sum((max_point - min_point) ** 2)) 
        pcd_normalized = pcd_centered / scale_factor
        return pcd_normalized

    def gaussian_pdf(self, x, mu, sigma):
        """ Compute Gaussian PDF. """
        num = mu.shape[0]
        cov = np.diag(sigma)
        part1 = 1 / ((2 * np.pi) ** (mu.shape[1] / 2) * (np.linalg.det(cov) ** 0.5))
        diff = x-mu
        diff = np.expand_dims(diff, axis=-1)
        inv_cov = np.linalg.inv(cov)
        inv_cov = np.tile(inv_cov, (num, 1, 1))
        middle_term = np.matmul(inv_cov, diff)
        quadratic_form = np.matmul(np.transpose(diff, axes=(0, 2, 1)), middle_term)
        part2 = np.exp(-0.5 * quadratic_form)
        return part1 * part2

    def sample_query(self, pcd, locative_noun):
        """ Sample a point cloud for a query """
        assert locative_noun in self.locative_nouns, "Invalid query. Available locative nouns: front, back, top, bottom, left, right, middle."
        # Normalize the point cloud
        pcd_normalized = self.normalize_pcd(pcd)
        # Get Gaussian parameters for the locative noun
        params = self.gaussian_param_mappings[locative_noun]
        mu, sigma = params['mu'], params['sigma']
        # Calculate the PDF values for each point
        num = pcd_normalized.shape[0]
        mu = np.array(mu).reshape((1,3))
        sigma = np.array(sigma)
        pdf_values = self.gaussian_pdf(pcd_normalized, np.tile(mu, (num, 1)), sigma)

        # Filter points based on PDF values
        max_pdf_value = np.max(pdf_values)
        # threshold = max_pdf_value / np.exp(1.5 ** 2)  # Corresponds to 1.5 std dev away in a Gaussian distribution
        
        std_pdf_value = np.std(pdf_values)
        threshold = max_pdf_value - 1.5 * std_pdf_value
        
        sampled_indices = np.where(pdf_values >= threshold)[0]
        sampled_pcd = pcd[sampled_indices]
        return sampled_pcd

def same_or_opposite(word1, word2):
    # Mapping of directions to their opposites
    opposites = {
        'front': 'back',
        'back': 'front',
        'top': 'bottom',
        'bottom': 'top',
        'left': 'right',
        'right': 'left'
    }

    # Check if the words are the same or opposites
    if word1 == word2 or opposites.get(word1) == word2:
        return True
    else:
        return False

def extract_skill_chain(chain_params):
    if len(chain_params)>1:
        skill_chain = "+".join(skill["skill_name"] for skill in chain_params)
    else:
        skill_chain = chain_params[0]["skill_name"]
    return skill_chain

def mapping_rotate(word):
    return {
        'front': 'front',
        'back': 'back',
        'top': 'upwards',
        'bottom': 'downwards',
        'left': 'left',
        'right': 'right'
    }.get(word)

def mapping_translate(word):
    return {
        'front': 'forwards',
        'back': 'backwards',
        'top': 'upwards',
        'bottom': 'downwards',
        'left': 'to the left',
        'right': 'to the right'
    }.get(word)