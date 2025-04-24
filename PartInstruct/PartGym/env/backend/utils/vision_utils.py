import open3d as o3d
import numpy as np
import trimesh
import scipy
import random
import cv2
import torch
import pytorch3d.ops as torch3d_ops

def compute_scaling_and_translation(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    
    # Compute the axis-aligned bounding box
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    
    # Calculate the main diagonal length of the bounding box
    bounding_box_diagonal = np.linalg.norm(max_corner - min_corner)
    
    # Determine the scaling factor to make the main diagonal length 1
    scaling_factor = 1 / bounding_box_diagonal
    
    # Calculate the center of the bounding box
    bbox_center = (max_corner + min_corner) / 2
    
    # Determine the translation vector to center the bounding box at (0,0,0)
    translation_vector = -bbox_center
    
    return scaling_factor, translation_vector

def depth_to_pcd(depth_image, intrinsic_matrix, depth_scale=1000.0, depth_trunc=3.0):
    """
    Convert a depth image to a point cloud using Open3D.

    Parameters:
    - depth_image: A numpy array representing the depth image.
    - intrinsic_matrix: 3x3 numpy array representing the camera intrinsic parameters.
    - depth_scale: The scale factor for the depth image (e.g., depth in millimeters to meters).
    - depth_trunc: The maximum depth value to retain (for truncating far objects).
    
    Returns:
    - point_cloud: Open3D PointCloud object containing the point cloud.
    """
    # Convert depth image to Open3D format
    depth_o3d = o3d.geometry.Image(depth_image)

    # Create an Open3D camera intrinsic object
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsics.intrinsic_matrix = intrinsic_matrix
    
    # Convert depth image to point cloud
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d,
        camera_intrinsics,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )

    return point_cloud

def downsample_pcd(pcd, downsample_size):
    pcd = pcd.reshape(-1, pcd.shape[-1])
    _, indices = torch3d_ops.sample_farthest_points(points=pcd.unsqueeze(0), K=downsample_size)
    return indices

def pad_point_cloud(pcd, target_size):
    current_size = pcd.shape[0]
    if current_size < target_size:
        # Get the last point of the current point cloud
        last_point = pcd[-1, :]
        padding = np.tile(last_point, (target_size - current_size, 1))
        pcd = np.vstack((pcd, padding))
    return pcd

def depth_to_pcd_farthest_points(depth_image, intrinsic_matrix, depth_scale=1.0, depth_trunc=3.0, height=300, width=300, downsample_size=1024, agent_view=False, device=None):
    # Convert depth image to Open3D format
    depth_o3d = o3d.geometry.Image(depth_image)

    # Create an Open3D camera intrinsic object
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsics.intrinsic_matrix = intrinsic_matrix
    
    # Convert depth image to point cloud
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d,
        camera_intrinsics,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
    )
    pcd = np.asarray(point_cloud.points)
    pcd_np = pcd.reshape(-1, pcd.shape[-1])
    pcd = torch.tensor(pcd_np, dtype=torch.float32).to(device)
    indices = downsample_pcd(pcd, downsample_size)        
    return pcd_np, indices

def sample_point_cloud_from_obj(obj_path, num_points):
    loaded = trimesh.load(obj_path)

    # Check if the loaded object is a Scene or a single Mesh
    if isinstance(loaded, trimesh.Scene):
        # If it's a Scene, attempt to convert it to a single Mesh
        # This works well if the Scene contains a single object or connected components
        if loaded.is_empty:
            raise ValueError("The OBJ file is empty or could not be loaded properly.")
        mesh = loaded.dump(concatenate=True)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError("The loaded object is neither a Scene nor a Mesh.")

    # Now that we have a single mesh, sample the point cloud
    point_cloud = mesh.sample(num_points)
    
    return point_cloud

def resample_pcd(pcd_np, target_num_points):
    current_num_points = len(pcd_np)
    
    if target_num_points == current_num_points:
        return pcd_np
    
    elif target_num_points < current_num_points:
        # Randomly sample indices to downsample
        sampled_indices = random.sample(range(current_num_points), target_num_points)
        downsampled_pcd_np = pcd_np[sampled_indices, :]
        return downsampled_pcd_np
    
    else:
        # Randomly duplicate points to upsample
        additional_points_needed = target_num_points - current_num_points
        duplicated_indices = np.random.choice(range(current_num_points), additional_points_needed)
        
        # Get the new points by selecting from the original points
        additional_points = pcd_np[duplicated_indices, :]
        
        # Combine the original points with the duplicated ones
        upsampled_pcd_np = np.vstack([pcd_np, additional_points])
        return upsampled_pcd_np

def connect_mask_points(mask, kernel_size=5, iterations=1):
    """
    Connects sparse points in a binary mask using morphological operations.
    
    Parameters:
    - mask: A binary mask of shape (height, width) where projected points are 1 and others are 0.
    - kernel_size: Size of the structuring element used for dilation and closing.
    - iterations: Number of iterations for the morphological operations.
    
    Returns:
    - processed_mask: The mask after applying morphological operations to connect points.
    """
    # Define the structuring element (kernel) for dilation/closing
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply dilation to expand the regions and connect nearby points
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)

    # Optionally apply closing (dilation followed by erosion) to connect sparse points and fill holes
    processed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return processed_mask

def calculate_camera_extrinsics(target, distance, yaw_deg, pitch_deg, up_axis=np.array([0, 0, 1])):
    # Convert angles to radians
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    
    # Calculate camera position
    cam_rel_pos = np.array([
        distance * np.cos(pitch) * np.cos(yaw),
        distance * np.cos(pitch) * np.sin(yaw),
        distance * np.sin(pitch)
    ])
    cam_pos = np.array(target) + cam_rel_pos
    
    # Calculate look-at direction
    forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
    
    # Calculate right vector
    print(forward, up_axis)
    right = np.cross(forward, up_axis)
    right = right / np.linalg.norm(right)
    
    # Recalculate up vector to ensure orthogonality
    up = np.cross(right, forward)
    
    # Construct view matrix
    view_matrix = np.eye(4)
    view_matrix[:3, :3] = np.column_stack((right, up, -forward))
    view_matrix[:3, 3] = -np.dot(view_matrix[:3, :3], cam_pos)
    
    return view_matrix

def project_point_cloud_to_mask(pcd, width, height, view_matrix, proj_matrix):
    """
    Projects a point cloud to 2D image coordinates and generates a binary mask.
    
    Parameters:
    - pcd: numpy array of shape (N, 3), where N is the number of points.
    - width, height: the dimensions of the output mask (image).
    - view_matrix: 4x4 view matrix from the camera in PyBullet.
    - proj_matrix: 4x4 projection matrix from the camera in PyBullet.
    
    Returns:
    - mask: A binary mask of shape (height, width) where projected points are 1 and others are 0.
    """
    # Convert matrices from PyBullet's list format to NumPy arrays
    view_matrix = np.array(view_matrix).reshape(4, 4, order="F")
    proj_matrix = np.array(proj_matrix).reshape(4, 4, order="F")

    # Convert point cloud to homogeneous coordinates (add 1s to the end)
    pcd_homogeneous = np.hstack((pcd, np.ones((pcd.shape[0], 1))))

    # Apply view transformation
    pcd_camera = pcd_homogeneous @ view_matrix.T

    # Apply projection transformation
    pcd_projected = pcd_camera @ proj_matrix.T

    # Normalize the homogeneous coordinates to get (x, y, z) in 2D space
    pcd_projected[:, 0] /= pcd_projected[:, 3]  # x / w
    pcd_projected[:, 1] /= pcd_projected[:, 3]  # y / w
    pcd_projected[:, 2] /= pcd_projected[:, 3]  # z / w (for depth)

    # Transform normalized coordinates to image coordinates
    x_img = ((pcd_projected[:, 0] + 1) * 0.5 * width).astype(int)
    y_img = ((1 - pcd_projected[:, 1]) * 0.5 * height).astype(int)  # Invert y axis for image space

    # Create the binary mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Mark the projected points as 1 in the mask if they fall within image bounds
    valid_indices = (x_img >= 0) & (x_img < width) & (y_img >= 0) & (y_img < height)
    mask[y_img[valid_indices], x_img[valid_indices]] = 1
    
    mask = connect_mask_points(mask, kernel_size=2, iterations=1)
    return mask

def quaternion_to_rotation_matrix(quat):
    """Convert a quaternion into a rotation matrix."""
    qx, qy, qz, qw = quat
    qx2, qy2, qz2 = qx * qx, qy * qy, qz * qz
    qwqx, qwqy, qwqz = qw * qx, qw * qy, qw * qz
    qxqy, qxqz, qyqz = qx * qy, qx * qz, qy * qz

    R = np.array([
        [1 - 2 * (qy2 + qz2),     2 * (qxqy - qwqz),     2 * (qxqz + qwqy)],
        [    2 * (qxqy + qwqz), 1 - 2 * (qx2 + qz2),     2 * (qyqz - qwqx)],
        [    2 * (qxqz - qwqy),     2 * (qyqz + qwqx), 1 - 2 * (qx2 + qy2)]
    ])
    return R

def transform_point_cloud(points, position, quaternion, scale=1.0):
    """Transform a point cloud given a target position and quaternion orientation.""" # T_world_obj T_world_pcd=I T_world_obj
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(quaternion)
    Scaled_R = R * scale
    # Create the transformation matrix
    transformation_matrix = np.eye(4)  # Initialize as identity matrix
    transformation_matrix[:3, :3] = Scaled_R  # Set rotation part
    transformation_matrix[:3, 3] = position  # Set translation part
    
    # Apply transformation
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))  # Convert to homogeneous coordinates 
    transformed_points_homogeneous = np.dot(points_homogeneous, transformation_matrix.T)
    transformed_points = transformed_points_homogeneous[:, :3]  # Convert back from homogeneous coordinates
    return transformed_points

def rotation_quaternion(fixed_axis, target_direction):
    # Normalize the vectors
    fixed_axis_normalized = fixed_axis / np.linalg.norm(fixed_axis)
    target_direction_normalized = target_direction / np.linalg.norm(target_direction)
    
    # Find the rotation axis (cross product)
    rotation_axis = np.cross(fixed_axis_normalized, target_direction_normalized)
    
    # Find the rotation angle (arc cosine of the dot product)
    cos_angle = np.dot(fixed_axis_normalized, target_direction_normalized)
    angle = np.arccos(cos_angle)
    
    # Normalize the rotation axis
    rotation_axis_normalized = rotation_axis / np.linalg.norm(rotation_axis)
    
    # Compute the quaternion components
    qx = rotation_axis_normalized[0] * np.sin(angle / 2)
    qy = rotation_axis_normalized[1] * np.sin(angle / 2)
    qz = rotation_axis_normalized[2] * np.sin(angle / 2)
    qw = np.cos(angle / 2)
    
    return np.array([qx, qy, qz, qw]), angle

def downsample_depth_map(depth_map, target_shape=(3, 100, 100)):
    # Use interpolation to downsample the depth map
    zoom_factors = (1, target_shape[1] / depth_map.shape[1], target_shape[2] / depth_map.shape[2])
    downsampled_depth_map = np.stack([scipy.ndimage.zoom(depth_map[channel], zoom_factors[1:], order=1) for channel in range(depth_map.shape[0])])
    return downsampled_depth_map

def visualize_point_cloud(points, color=[1, 0, 0], draw_axes=False):
    """
    Visualize a point cloud with an optional coordinate axes using Open3D.

    Parameters:
    - points: numpy.ndarray of shape (N, 3), points of the point cloud.
    - color: The RGB color of the point cloud as a list [R, G, B].
    - draw_axes: Boolean indicating whether to draw coordinate axes.
    """
    # Create the point cloud and assign a color
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.array([color for _ in range(len(points))]))

    geometries = [pcd]

    if draw_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(axes)

    # Visualize the point cloud
    o3d.visualization.draw_geometries(geometries)

def visualize_segmentation_mask(segmentation_mask):
    # Define a colormap
    colormap = {
        0: (255, 255, 255),       # Background (black)
        1: (255, 0, 0),     # Class 1 (red)
        2: (0, 255, 0),     # Class 2 (green)
        3: (0, 0, 255),     # Class 3 (blue)
        4: (255, 255, 0),   # Class 4 (yellow)
        5: (255, 165, 0),   # Class 5 (orange)
        6: (128, 0, 128),   # Class 6 (purple)
        7: (0, 255, 255),   # Class 7 (cyan)
    }
    
    # Create an empty RGB image with the same height and width as the segmentation mask
    height, width = segmentation_mask.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Map each unique value in the segmentation mask to the corresponding color
    for class_id, color in colormap.items():
        color_image[segmentation_mask == class_id] = color
    
    return color_image

def apply_segmentation_mask(depth_map, segmentation_mask):
    """
    Apply a segmentation mask to a depth map. Background pixels (mask == 0) will be set to 0.
    
    Parameters:
    - depth_map: A 2D numpy array representing the depth map.
    - segmentation_mask: A 2D numpy array representing the segmentation mask (same size as depth_map).
                         Pixels with value 0 are treated as background, non-zero as foreground.
    
    Returns:
    - masked_depth: A 2D numpy array where background pixels are set to 0 and
                    foreground pixels retain their original depth.
    """
    # Ensure the segmentation mask is a binary mask (foreground: True, background: False)
    foreground_mask = segmentation_mask != 0
    
    # Apply the mask to the depth map, setting background depth values to 0
    masked_depth = np.where(foreground_mask, depth_map, 0)
    
    return masked_depth

def visualize_and_save_two_point_clouds(points1, points2, color1=[1, 0, 0], color2=[0, 1, 0], save_path="pcd.ply", visualize=True):
    """
    Visualize two point clouds with different colors using Open3D, and optionally save them as a .ply file.
    
    Parameters:
    - points1: numpy.ndarray of shape (N, 3), points of the first point cloud.
    - points2: numpy.ndarray of shape (M, 3), points of the second point cloud.
    - color1: The RGB color of the first point cloud as a list [R, G, B].
    - color2: The RGB color of the second point cloud as a list [R, G, B].
    - save_path: Optional. If provided, the function will save the combined point cloud as a .ply file.
    """
    # Create the first point cloud and assign a color
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd1.colors = o3d.utility.Vector3dVector(np.array([color1 for _ in range(len(points1))]))

    # Create the second point cloud and assign a color
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd2.colors = o3d.utility.Vector3dVector(np.array([color2 for _ in range(len(points2))]))

    # Combine the two point clouds
    combined_pcd = pcd1 + pcd2

    if visualize:
        # Visualize the point clouds
        o3d.visualization.draw_geometries([combined_pcd])

    # Optionally save the combined point cloud as a .ply file
    if save_path is not None:
        o3d.io.write_point_cloud(save_path, combined_pcd)
        print(f"Combined point cloud saved as {save_path}")

def save_pcd(points, file_path):
    """
    Converts world coordinates to a point cloud and saves it as a PCD file.

    Parameters:
    - points: numpy array of shape (N, 3)
    - file_path: string, path to save the PCD file
    """
    # Flatten the points
    points = points.reshape(-1, 3)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Save the point cloud to a PCD file
    o3d.io.write_point_cloud(file_path, pcd)
    print(f"Point cloud saved to {file_path}")

def sigmoid(x, scale=1.0):
    """Apply a sigmoid transformation to x with an optional scaling factor."""
    return 1 / (1 + np.exp(-x / scale))