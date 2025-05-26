from PartInstruct.PartGym.env.backend.utils.transform import *
import json
from pathlib import Path
import random

def load_scene_objects(world, config_path, objects_directory):
        # Read the JSON configuration file
        with open(config_path, 'r') as file:
            config = json.load(file)
        
        # Iterate through each object instance in the JSON configuration T_world_w2=T_world_scene*T_w2_scene.inverse() 
        for obj_instance in config["object_instances"]:
            template_name = obj_instance["template_name"]
            translation = obj_instance["translation"]
            rotation = obj_instance["rotation"]
            
            # Construct the path to the URDF file
            urdf_path = Path(objects_directory) / template_name / f"{template_name}.urdf"
            
            # Check if the URDF file exists
            if urdf_path.exists():
                # Construct the pose
                reflection = Rotation.from_matrix(np.array([[-1, 0, 0],
                                                            [0, -1, 0],
                                                            [0, 0, -1]]))
                T_world_obj = Transform(Rotation.identity(), [-1.9, 5.62, 0.12])*Transform(Rotation.from_euler('xyz', [90, 0, 0], degrees=True), [0.0, 0.0, 0.0])*Transform(reflection*Rotation(rotation), translation)

                world.load_urdf(urdf_path, T_world_obj)
            else:
                print(f"URDF file not found for {template_name} at {urdf_path}")
        # step the simulation to let the object settle
        for _ in range(60):
            world.step()

def load_replica_cad_scene(world, scene_path=None, scene_config_path=None, objects_directory=None):
    scene = None
    if scene_path is not None:
        urdf_scene = scene_path
        reflection = Rotation.from_matrix(np.array([[-1, 0, 0],
                                                    [0, -1, 0],
                                                    [0, 0, -1]]))
        T_world_scene = Transform(Rotation.identity(), [-1.0, 4.0, 1.55])*Transform(reflection*Rotation([1, 0, 0, 0]), [0.0, 0.0, 0.0])*Transform(Rotation.from_euler('xyz', [90, 0, 0], degrees=True), [0.0, 0.0, 0.0])
    
        scene = world.load_urdf(urdf_scene, T_world_scene, scale=1.0, useFixedBase=True)
        if scene_config_path is not None:
            load_scene_objects(world, scene_config_path, objects_directory)
    return scene

def wait_for_objects_to_rest(world, obj, timeout=2.0, tol=0.01):
        timeout = world.sim_time + timeout
        objects_resting = False
        while not objects_resting and world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                world.step()
            # check whether all objects are resting
            objects_resting = True
            
            if np.linalg.norm(obj.get_velocity()) > tol:
                objects_resting = False
                break

def add_variation_to_position(position, variation=0.025):
    varied_position = [
        pos + random.uniform(-variation, variation) for pos in position
    ]
    return varied_position

def load_grasping_setups(world, urdf_floor, urdf_table):
    # place floor
    pose = Transform(Rotation.from_euler('xyz', [90, 0, 90], degrees=True), [0.15, 0.05, 0.05+0.01])
    floor = world.load_urdf(urdf_floor, pose, scale=1.0, useFixedBase=True)
    # place table
    pose = Transform(Rotation.from_euler('xyz', [90, 0, 90], degrees=True), [0.15, 0.05, 0.05+0.012])
    table = world.load_urdf(urdf_table, pose, scale=0.5, useFixedBase=True)
    return table, floor

def load_grasping_object(world, urdf_obj, obj_position=None, obj_orientation=None, obj_scale=1.0):
    # load objects
    if obj_position is None:
        obj_position = [0.0, -0.031480762319536344, 0.2]
        obj_position = add_variation_to_position(obj_position)
    obj_position[2] = 0.15
        
    if obj_orientation is None:
        obj_orientation = list(Rotation.identity().as_quat())
    
    obj_rotation = Rotation.from_quat(obj_orientation)
    
    pose = Transform(obj_rotation*Rotation.from_euler('xyz', [90, 0, 0], degrees=True), np.array(obj_position))
    
    if "mug" in str(urdf_obj):
        obj = world.load_urdf(urdf_obj, pose, scale=obj_scale, flags=world.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    else:
        obj = world.load_urdf(urdf_obj, pose, scale=obj_scale, flags=world.p.URDF_USE_MATERIAL_COLORS_FROM_MTL)

    wait_for_objects_to_rest(world, obj, timeout=2.0)
    return obj

def make_invisible(world, object_id):
    """
    Make the object with the given object_id invisible by setting its color's alpha value to 0.
    
    :param object_id: The PyBullet ID of the object to make invisible.
    """
    # Get visual shape data for the object
    visual_shape_data = world.p.getVisualShapeData(object_id)
    
    # Loop over all visual shapes and make them invisible
    for visual in visual_shape_data:
        object_unique_id = visual[0]
        link_index = visual[1]

        # Set the color to fully transparent (alpha = 0)
        world.p.changeVisualShape(object_unique_id, link_index, rgbaColor=[1, 1, 1, 0])

def safe_get(lst, index):
    try:
        return lst[index]
    except IndexError:
        return None