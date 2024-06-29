import open3d as o3d
import json
import numpy as np
import cv2
import os # for absolute path for idiots like me



def pixel_coordinates_to_angles(u, v, intrinsics):
    u -= 1920/2
    v -= 1440/2
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]

    angle_around_x = -np.arctan2(v, fy)
    angle_around_y = -np.arctan2(u, fx)
    
    return angle_around_x, angle_around_y



def rotate_vector(to_rotate, axis, radians):
    # Normalize the input vectors
    to_rotate = to_rotate / np.linalg.norm(to_rotate)
    axis = axis / np.linalg.norm(axis)
    
    # Calculate components of Rodrigues' rotation formula
    cos_theta = np.cos(radians)
    sin_theta = np.sin(radians)
    dot_product = np.dot(axis, to_rotate)
    cross_product = np.cross(axis, to_rotate)
    
    # Apply Rodrigues' rotation formula
    rotated_vector = (to_rotate * cos_theta +
                      cross_product * sin_theta +
                      axis * dot_product * (1 - cos_theta))
    
    return rotated_vector



def add_spheres_along_line(pcd, extrinsics, ray, radius):

    #Extract all vertices from the pointcloud
    vertices = np.asarray(pcd.points)
    line_direction = ray / np.linalg.norm(ray)

    # Find points within the specified radius of the line defined by direction and camera location
    closest_points = extrinsics[:3, 3] + np.dot((vertices - extrinsics[:3, 3]), line_direction)[:, np.newaxis] * line_direction
    distances = np.linalg.norm(vertices - closest_points, axis=1)
    highlighted_indices = np.where(distances <= radius)[0]
    distances_to_camera = np.linalg.norm(vertices[highlighted_indices] - extrinsics[:3, 3], axis=1)

    # The closest index will be selected
    closest_index = highlighted_indices[np.argmin(distances_to_camera)]
    
    # Place spheres on all detected points
    spheres = []
    for idx in highlighted_indices:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        if idx == closest_index:
            sphere.paint_uniform_color([0, 1, 0]) # The closest point will be green
            print(vertices[idx])
        else:
            sphere.paint_uniform_color([1, 0, 0]) # All other points will be red
        sphere.translate(vertices[idx])
        spheres.append(sphere)

    return spheres, closest_index



def show_pointcloud_frame(id, x, y):
    
    # Import data from the frame
    # with open(f'../scan/data/{id}.json') as f:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f'{script_dir}\..\scan\data\{id}.json') as f: # absolute path for idiots like me
        frame_data = json.load(f)
    intrinsics = np.reshape(np.array(frame_data['intrinsics']), (3, 3))
    extrinsics = np.reshape(np.array(frame_data['cameraPoseARFrame']), (4, 4))

    # Create visualizer window and add point cloud of environment
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1440)
    # pcd = o3d.io.read_point_cloud("../scan/pointcloud.ply")
    pcd = o3d.io.read_point_cloud(f'{script_dir}\..\scan\pointcloud.ply')
    vis.add_geometry(pcd)

    # Extract camera view and rotation from extrinsics
    rotation_matrix = extrinsics[:3, :3]
    camera_position = extrinsics[:3, 3]

    # Camera z-direction
    z_axis_camera_reference = -rotation_matrix[:, 2]
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([camera_position, camera_position + z_axis_camera_reference])
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
    vis.add_geometry(line)
    
    # Camera x-direction
    x_axis_camera_reference = rotation_matrix[:, 0]
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([camera_position, camera_position + x_axis_camera_reference])
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
    vis.add_geometry(line)

    # Camera y-direction
    y_axis_camera_reference = rotation_matrix[:, 1]
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([camera_position, camera_position + y_axis_camera_reference])
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
    vis.add_geometry(line)

    # Get angles for selecting direction
    angle_around_x, angle_around_y = pixel_coordinates_to_angles(x, y, intrinsics)

    # Selecting direction
    selecting_direction = z_axis_camera_reference
    selecting_direction = rotate_vector(to_rotate=selecting_direction, axis=x_axis_camera_reference, radians=angle_around_x)
    selecting_direction = rotate_vector(to_rotate=selecting_direction, axis=y_axis_camera_reference, radians=angle_around_y)
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector([camera_position, camera_position + selecting_direction * 3])
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.colors = o3d.utility.Vector3dVector([[1, 0, 1]])
    vis.add_geometry(line)

    # Add spheres along selecting direction and find the pointcloud indext with the closest distance to the camera
    spheres, closest_index = add_spheres_along_line(pcd, extrinsics, selecting_direction, radius=0.01)
    for sphere in spheres:
        vis.add_geometry(sphere)

    # Add a dice mesh add the selected 3D point
    # dice_mesh = o3d.io.read_triangle_mesh("../hologram_models/dice.stl")
    dice_mesh = o3d.io.read_triangle_mesh(f"{script_dir}\..\hologram_models\dice.stl")
    vertices = np.asarray(pcd.points)
    dice_point = vertices[closest_index]
    dice_mesh.translate(dice_point, relative=False) # Move dice to selected 3D point
    dice_mesh.compute_vertex_normals() # Add shading to dice
    dice_mesh.scale(0.004, center=dice_point)
    R = dice_mesh.get_rotation_matrix_from_xyz((-np.pi/2, 0, 0))
    dice_mesh.rotate(R, center=dice_point) # Rotate dice to match the pointcloud

    # Get y-axis rotation of selecting_direction to rotate the dice; because y-axis is the up vector
    global_z = np.array([0, 0, 1])
    global_y = np.array([0, 1, 0])
    global_x = np.array([1, 0, 0])
    selecting_direction = selecting_direction / np.linalg.norm(selecting_direction)
    selecting_direction_xz = selecting_direction - np.dot(selecting_direction, global_y) * global_y # Project the selecting direction to the xz-plane
    selecting_direction_xz = selecting_direction_xz / np.linalg.norm(selecting_direction_xz)
    rotation_angle = np.arctan2(np.cross(global_x, selecting_direction_xz), np.dot(global_x, selecting_direction_xz)) # Angle between x-axis and selecting direction in xz-plane
    rotation_angle[1] -= np.pi/2 # Rotation around y-axis is 90 degrees off

    # Rotate dice aroung global z-axis to face the selecting direction
    R = dice_mesh.get_rotation_matrix_from_xyz((0, rotation_angle[1], 0)) # Rotate around y-axis; ignore x and z values
    dice_mesh.rotate(R, center=dice_point)
    
    # Rotate the dice to face the selecting direction; somehow rotates in the same direction as vieweing angle in the pointcloud
    # rotation_matrix_dice = [x_axis_camera_reference, y_axis_camera_reference, z_axis_camera_reference]
    # dice_mesh.rotate(rotation_matrix_dice, center=dice_point) # creepy, spooky and scary
    # vis.add_geometry(dice_mesh)

    # Camera position
    camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
    camera_sphere.paint_uniform_color([0, 0, 1])
    camera_sphere.translate(extrinsics[:3, 3])
    vis.add_geometry(camera_sphere)

    # Correct camera view to world frame and set camera parameters
    inverted_extrinsics = np.linalg.inv(extrinsics)
    rotation_x = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])
    corrected_inverted_extrinsics = np.dot(rotation_x, inverted_extrinsics)
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    params.intrinsic.set_intrinsics(1920, 1440, intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
    params.extrinsic = corrected_inverted_extrinsics
    ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)

    # Save an image of the 3D view
    image = vis.capture_screen_float_buffer(True)
    # o3d.io.write_image(f"../outputs/{id}_pcd_view.png", o3d.geometry.Image((np.array(image) * 255).astype(np.uint8)))
    o3d.io.write_image(f"{script_dir}\..\outputs\{id}_pcd_view.png", o3d.geometry.Image((np.array(image) * 255).astype(np.uint8)))

    # Display the 3D environment
    vis.update_renderer()
    vis.run()
    vis.destroy_window()



def brisk_frame(id):

    # Read image corresponding to the frame id
    # img = cv2.imread(f'../scan/data/{id}.jpg')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img = cv2.imread(f'{script_dir}\..\scan\data\{id}.jpg') # windows with the fucking backslash

    # Perform BRISK feature extraction
    brisk = cv2.BRISK_create()
    keypoints = brisk.detect(img, None)

    # Filter keypoints for the most relevant ones and hold minimum distance between them
    max_keypoints = 1000
    min_distance = 20
    filtered_keypoints = []
    for kp in keypoints:
        if all(np.linalg.norm(np.array(kp.pt) - np.array(k.pt)) > min_distance for k in filtered_keypoints):
            filtered_keypoints.append(kp)
        if len(filtered_keypoints) >= max_keypoints:
            break

    # Callback function for mouseclick
    def mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:

            # Find the keypoint closest to the click
            closest_keypoint = min(filtered_keypoints, key=lambda kp: np.linalg.norm(np.array(kp.pt) - np.array((x, y))))
            print("Closest keypoint:", closest_keypoint.pt)

            # Highlight the selected keypoint and display with other keypoints
            img_with_circle = img_with_keypoints.copy()
            cv2.circle(img_with_circle, (int(closest_keypoint.pt[0]), int(closest_keypoint.pt[1])), 10, (255, 0, 0), -1)
            cv2.imshow("BRISK Keypoints", img_with_circle)
            show_pointcloud_frame(id, closest_keypoint.pt[0], closest_keypoint.pt[1])

    # Show image and draw all keypoints
    img_with_keypoints = cv2.drawKeypoints(img, filtered_keypoints, None, color=(0, 0, 255))
    img_with_filled_keypoints = img_with_keypoints.copy()
    # for kp in filtered_keypoints: # Draw filled circles around keypoints
    #     cv2.circle(img_with_filled_keypoints, (int(kp.pt[0]), int(kp.pt[1])), 10, (0, 0, 255), -1)
    cv2.imshow("BRISK Keypoints", img_with_filled_keypoints)
    cv2.setMouseCallback("BRISK Keypoints", mouse_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def main():
    id = 'frame_00064'
    print(f"Executing script for {id}")
    brisk_frame(id)



if __name__ == "__main__":
    main()
