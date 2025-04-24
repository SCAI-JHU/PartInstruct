import os, sys
import numpy as np
import open3d as o3d
import json
import shutil
import random

def process_dataset(src_root_dir, dst_root_dir):
    # Create a dictionary to store object names and their corresponding folders
    obj_folders = {}

    # Iterate through all folders in the source root directory
    for obj_id in os.listdir(src_root_dir):
        print(f"Processing {obj_id} ...")
        src_obj_dir = os.path.join(src_root_dir, obj_id)
        if os.path.isdir(src_obj_dir):
            # Load the result_after_merging.json file
            json_file = os.path.join(src_obj_dir, "result_after_merging.json")
            if not os.path.exists(json_file):
                json_file = os.path.join(src_obj_dir, "result.json")
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data = json.load(f)

                # Get the object name from the first node in the hierarchy tree
                obj_name = data[0]["name"]
                obj_name = obj_name.lower()

                # Check if a folder for this object name already exists
                if obj_name not in obj_folders:
                    # Create a new folder for the object name in the destination root directory
                    obj_folder = os.path.join(dst_root_dir, obj_name)
                    os.makedirs(obj_folder, exist_ok=True)
                    obj_folders[obj_name] = obj_folder

                # Copy the object ID folder to the corresponding object name folder in the destination root directory
                dst_obj_dir = os.path.join(obj_folders[obj_name], obj_id)
                shutil.copytree(src_obj_dir, dst_obj_dir)

    print("Dataset processing completed.")

def generate_urdf_from_json(json_path, urdf_path, use_color=True):
    with open(json_path) as f:
        data = json.load(f)

    def random_color():
        return f"{random.random()} {random.random()} {random.random()} 1"

    urdf_content = '<?xml version="1.0" ?>\n\n<robot name="mobility.urdf">\n'
    
    color = random_color()
    friction_coefficient = "1.0"  # Adjust this value as necessary
    if use_color:
      color_content = f"""<material name="{color}">
        <color rgba="{color}"/>
      </material>\n"""
    else:
      color_content = ""

    if data:
        objs = data[0].get('objs', [])
        if objs:
            # Base part
            base_obj = objs[0]
            urdf_content += f"""  <link concave="yes" name="base_part">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="textured_objs/{base_obj}.obj"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0.00001 0 0"/>
      {color_content}
    </visual>
    <collision concave="yes">
      <geometry>
        <mesh filename="textured_objs/{base_obj}.obj"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0.00001 0 0"/>
      <surface>
        <friction>
          <ode mu="{friction_coefficient}" mu2="{friction_coefficient}"/>
        </friction>
      </surface>
    </collision>
  </link>\n"""
            
            # Other parts
            for obj in objs[1:]:
                urdf_content += f"""  <link concave="yes" name="{obj}_part">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="textured_objs/{obj}.obj"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0.00001 0 0"/>
      {color_content}
    </visual>
    <collision concave="yes">
      <geometry>
        <mesh filename="textured_objs/{obj}.obj"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0.00001 0 0"/>
      <surface>
        <friction>
          <ode mu="{friction_coefficient}" mu2="{friction_coefficient}"/>
        </friction>
      </surface>
    </collision>
  </link>\n"""
                
                urdf_content += f"""  <joint name="joint_base_to_{obj}" type="fixed">
    <parent link="base_part"/>
    <child link="{obj}_part"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>\n"""

    urdf_content += '</robot>'
    
    with open(urdf_path, 'w') as f:
        f.write(urdf_content)

def generate_urdf_from_json2(json_path, urdf_path, use_color=False):
    with open(json_path) as f:
        data = json.load(f)
      
    def random_color():
        return f"{random.random()} {random.random()} {random.random()} 1"
    if use_color:
      color = random_color()
    else:
      color = None

    urdf_content = '<?xml version="1.0" ?>\n\n<robot name="mobility.urdf">\n'
    friction_coefficient = "1.0" 

    # Initialize base part content with inertial properties
    base_part_visuals = ""
    base_part_collisions = ""
    base_part_inertial = """  <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>\n"""

    if data:
        for item in data:
            children = item.get('children', [])
            for child in children:
                if child['name'] == "base_body":
                    # Accumulate visuals and collisions for base_body
                    base_body_objs = child.get('objs', [])
                    for obj in base_body_objs:
                        base_part_visuals += generate_visual(obj, color)
                        base_part_collisions += generate_collision(obj, friction_coefficient)
                else:
                    # Handle other parts
                    other_objs = child.get('objs', [])
                    for obj in other_objs:
                        urdf_content += generate_link(obj, friction_coefficient, color)
                        urdf_content += generate_joint(obj)
    urdf_content += f"""  <link concave="yes" name="base_part">
  {base_part_inertial}
  {base_part_visuals}
  {base_part_collisions}
    </link>\n"""

    urdf_content += '</robot>'
    
    with open(urdf_path, 'w') as f:
        f.write(urdf_content)

def generate_visual(obj_name, color=None):
    if color is not None:
        color_content = f"""<material name="{color}">
        <color rgba="{color}"/>
      </material>\n"""
    else:
        color_content = ""
    return f"""    <visual>
      <geometry>
        <mesh filename="textured_objs/{obj_name}.obj"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0.00001 0 0"/>
      {color_content}
    </visual>\n"""

def generate_collision(obj_name, friction_coefficient):
    return f"""    <collision concave="yes">
      <geometry>
        <mesh filename="textured_objs/{obj_name}.obj"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0.00001 0 0"/>
      <surface>
        <friction>
          <ode mu="{friction_coefficient}" mu2="{friction_coefficient}"/>
        </friction>
      </surface>
    </collision>\n"""

def generate_link(obj_name, friction_coefficient, color):
    link_name = f"{obj_name}_part"
    return f"""  <link concave="yes" name="{link_name}">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
{generate_visual(obj_name, color)}
{generate_collision(obj_name, friction_coefficient)}
  </link>\n"""

def generate_joint(obj_name):
    return f"""  <joint name="joint_base_to_{obj_name}" type="fixed">
    <parent link="base_part"/>
    <child link="{obj_name}_part"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>\n"""

def walk_and_generate_urdfs(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # if file == 'result.json':
            if file == 'result_after_merging.json':
                json_path = os.path.join(root, file)
                urdf_path = os.path.join(root, "mobility.urdf")
                generate_urdf_from_json(json_path, urdf_path, use_color=False)
                # generate_urdf_from_json2(json_path, urdf_path)
                print(f"Generated URDF for {json_path}")

# Replace 'root_directory_path' with the path to your dataset root
root_directory_path = '/media/yyin34/ExtremePro/projects/language-guided-manipulation/lgplm_sim/data/test'
for dir in os.listdir(root_directory_path):
  walk_and_generate_urdfs(os.path.join(root_directory_path, dir))

# src_dataset_root = "/media/yyin34/ExtremePro/projects/language-guided-manipulation/lgm_ws/lgplm_sim/lgplm_sim/data/partnet-mobility"

# # Specify the destination root directory for the processed dataset
# dst_dataset_root = "/media/yyin34/ExtremePro/projects/language-guided-manipulation/lgm_ws/lgplm_sim/lgplm_sim/data/partnet-mobility-classified"

# # Process the dataset
# process_dataset(src_dataset_root, dst_dataset_root)