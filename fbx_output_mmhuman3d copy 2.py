import bpy
import numpy as np
from mathutils import Matrix, Vector, Quaternion, Euler
import os.path
# Globals
male_model_path = 'C:/Users/USER/Desktop/SMPL_unity_v.1.0.0/SMPL_unity_v.1.0.0/smpl/Models/SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx'
female_model_path = 'C:/Users/USER/Desktop/SMPL_unity_v.1.0.0/SMPL_unity_v.1.0.0/smpl/Models/SMPL_f_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx'

bone_name_from_index = {
    0: 'Pelvis',
    1: 'L_Hip',
    2: 'R_Hip',
    3: 'Spine1',
    4: 'L_Knee',
    5: 'R_Knee',
    6: 'Spine2',
    7: 'L_Ankle',
    8: 'R_Ankle',
    9: 'Spine3',
    10: 'L_Foot',
    11: 'R_Foot',
    12: 'Neck',
    13: 'L_Collar',
    14: 'R_Collar',
    15: 'Head',
    16: 'L_Shoulder',
    17: 'R_Shoulder',
    18: 'L_Elbow',
    19: 'R_Elbow',
    20: 'L_Wrist',
    21: 'R_Wrist',
    22: 'L_Hand',
    23: 'R_Hand'
}

# Computes rotation matrix through Rodrigues formula
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat

# Setup scene
def setup_scene(model_path, fps_target, person_ids):
    scene = bpy.context.scene

    # Set up a dictionary to keep track of armature objects for each person_id
    armatures = {}

    # Remove default cube
    if 'Cube' in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()

    # Import gender-specific .fbx template file
    for person_id in range(person_ids):
        # Append a unique identifier to the armature name based on person_id
        armature_name = f'Armature_{person_id}'
        bpy.ops.import_scene.fbx(filepath=model_path)

        # Rename the imported armature object
        armature = bpy.context.active_object
        armature.name = armature_name
        armatures[person_id] = armature
        armature.location = Vector((0, 0, person_id * 2))

    # Now, armatures is a dictionary where keys are person_ids and values are corresponding armature objects
    return armatures

# Process single pose into keyframed bone orientations
def process_pose(current_frame, pose, trans, person_id_, source_armature):
    mat_rots = [Rodrigues(rod_rot) for rod_rot in pose]

    print(f"Processing pose for person_id {person_id_} at frame {current_frame}")

    armature = source_armature
    bones = armature.pose.bones
    bones[bone_name_from_index[0]].location = Vector((100 * trans[1], 100 * trans[2], 100 * trans[0]))  # - pelvis_position
    bones['f_avg_root'].location = Vector(trans)
    bones[bone_name_from_index[0]].keyframe_insert('location', frame=current_frame)

    for index, mat_rot in enumerate(mat_rots, 0):
        if index >= 24:
            continue

        bone = bones[bone_name_from_index[index]]
        bone_rotation = Matrix(mat_rot).to_quaternion()

        quat_x_90_cw = Quaternion((1.0, 0.0, 0.0), np.radians(-90))
        quat_z_180_cw = Quaternion((0.0, 0.0, 1.0), np.radians(-180))

        if index == 0:
            # Rotate pelvis so that the avatar stands upright and looks along the negative Y axis
            bone.rotation_quaternion = (quat_x_90_cw @ quat_z_180_cw) @ bone_rotation
        else:
            bone.rotation_quaternion = bone_rotation

        bone.keyframe_insert('rotation_quaternion', frame=current_frame)

    return

# Process all the poses from the pose file
def process_poses(input_path, gender, fps_source, fps_target, start_origin, person_id=1):
    print('Processing: ' + input_path)

    file_data = np.load(input_path, allow_pickle=True)

    body_pose = file_data['smpl'][()]['body_pose']
    global_orient = file_data['smpl'][()]['global_orient'].reshape(-1, 1, 3)
    pred_cams = file_data['pred_cams'].reshape(-1, 3)
    person_ids = file_data['person_id']
    # keypoints_3d = file_data['keypoints_3d'][()]

    poses = []
    poses1 = []
    poses2 =[]
    trans = []
    # for i in range(body_pose.shape[0]):
    #     if person_ids[i] == 1:
    #         poses1.append(np.concatenate([global_orient[i], body_pose[i]], axis=0))
    #     elif person_ids[i] == 2:
    #         poses2.append(np.concatenate([global_orient[i], body_pose[i]], axis=0))
    #     else:
    #         poses.append(np.concatenate([global_orient[i], body_pose[i]], axis=0))
    for i in range(body_pose.shape[0]):
        poses.append(np.concatenate([global_orient[i], body_pose[i]], axis=0))
    #     #trans.append((keypoints_3d[i][8] + keypoints_3d[i][11])/2 * 5)
        trans.append(pred_cams[i])

    poses = np.array(poses)
    poses1 = np.array(poses1)
    poses2 = np.array(poses2)
    trans = np.array(trans)

    # trans = np.zeros((poses.shape[0], 3))

    if gender == 'female':
        model_path = female_model_path
        for k, v in bone_name_from_index.items():
            bone_name_from_index[k] = 'f_avg_' + v
           
    elif gender == 'male':
        model_path = male_model_path
        for k, v in bone_name_from_index.items():
            bone_name_from_index[k] = 'm_avg_' + v
            
    else:
        print('ERROR: Unsupported gender: ' + gender)
        sys.exit(1)

    # Limit target fps to source fps
    if fps_target > fps_source:
        fps_target = fps_source

    print(f'Gender: {gender}')
    print(f'Number of source poses: {str(poses.shape[0])}')
    print(f'Source frames-per-second: {str(fps_source)}')
    print(f'Target frames-per-second: {str(fps_target)}')
    print('--------------------------------------------------')

    armatures = setup_scene(model_path, fps_target, max(set(person_ids)) + 1)

    scene = bpy.data.scenes['Scene']
    sample_rate = int(fps_source / fps_target)
    scene.frame_end = (int)(poses.shape[0] / sample_rate)

    # Retrieve pelvis world position.
    # Unit is [cm] due to Armature scaling.
    # Need to make copy since reference will change when bone location is modified.
    bpy.ops.object.mode_set(mode='EDIT')
    #pelvis_position = Vector(bpy.data.armatures[0].edit_bones[bone_name_from_index[0]].head)
    bpy.ops.object.mode_set(mode='OBJECT')

    source_index = 0
    frame = 1
    
    offset = np.array([0.0, 0.0, 0.0])

    while source_index < poses.shape[0]:
        print('Adding pose: ' + str(source_index))

        if start_origin:
            if source_index == 0:
                offset = np.array([trans[source_index][0], trans[source_index][1], 0])
        
    # Go to new frame
        scene.frame_set(frame)
        if person_ids[source_index] == 6:
            source_index += sample_rate
            frame += 1
            continue
        
        armature = armatures[person_ids[source_index]]
        
            
        process_pose(frame, poses[source_index], (trans[source_index] - offset), person_ids[source_index],armature)
        source_index += sample_rate
        frame += 1
                
             
    return frame
               
if __name__ == '__main__':
    input_path = 'C:/Users/USER/Desktop/SMPL_unity_v.1.0.0\inference_result_multi10.npz'
    fps_source = 30
    fps_target = 30
    gender = 'female'
    start_origin = 1
    person_id = 0

    # Process data
    cwd = bpy.path.abspath('//')

    # Turn relative input path into an absolute path
    if not input_path.startswith('//'):
        input_path = bpy.path.abspath(input_path)

    print('Input path: ' + input_path)

    # Process pose file
    if input_path.endswith('.npz'):
        if not os.path.isfile(input_path):
            print('ERROR: Invalid input file')
            bpy.ops.wm.quit_blender()
        process_poses(
            input_path=input_path,
            gender=gender,
            fps_source=fps_source,
            fps_target=fps_target,
            start_origin=start_origin,
            person_id=person_id
        )

    print('--------------------------------------------------')
    print('Animation import finished.')
    print('--------------------------------------------------')
