import pathlib

### Task parameters
DATA_DIR = '/home/kist/data/aloha'
SIM_TASK_CONFIGS = {

# ------------------original version ----------------------------

    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_transfer_cube_scripted_angle':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle']
    },

    'sim_insertion_scripted_angle': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle']
    },

    'sim_transfer_cube_scripted_act':{
        'dataset_dir': '/home/kist/data/sim/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted_act': {
        'dataset_dir': '/home/kist/data/sim/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_transfer_cube_human_act':{
        'dataset_dir': '/home/kist/data/sim/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_human_act': {
        'dataset_dir': '/home/kist/data/sim/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },


    'sim_transfer_cube_record':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle']
    },

    'sim_cube_classification_record':{
        'dataset_dir': DATA_DIR + '/sim_cube_classification_scripted',
        'num_episodes': 50,
        'episode_len': 1700,
        'camera_names': ['top', 'angle', 'angle_down', "left_pillar", "right_pillar",'left_wrist', 'right_wrist']
    },

    'sim_insertion_record':{
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle']
    },
#------------------3D environment recording ---------------------------

    'sim_transfer_cube_3d_record':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'angle', 'angle_down', 'left_pillar', 'right_pillar', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_3d_record':{
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'angle', 'angle_down', 'left_pillar', 'right_pillar', 'left_wrist', 'right_wrist']
    },

#------------------3D environment: single camera -----------------------

    'sim_transfer_cube_3d_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_3d_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_transfer_cube_3d_scripted_angle':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle']
    },

    'sim_insertion_3d_scripted_angle': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle']
    },

#------------------3D environment: gripper only --------------------------

    'sim_transfer_cube_3d_scripted_gripper_only':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_wrist', 'right_wrist']
    },

    'sim_insertion_3d_scripted_gripper_only': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_wrist', 'right_wrist']
    },

#------------------3D environment: top + angle (2 cams) ------------------

    'sim_transfer_cube_3d_scripted_top_angle':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'angle']
    },

    'sim_insertion_3d_scripted_top_angle': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'angle']
    },

#------------------3D environment: two eyes (horizontal) ----------------

    'sim_transfer_cube_3d_scripted_twoeyes':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar']
    },

    'sim_insertion_3d_scripted_twoeyes': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar']
    },

    'sim_transfer_cube_3d_scripted_cross':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar']
    },

    'sim_insertion_3d_scripted_cross': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar']
    },

#------------------3D environment: two eyes + gripper -------------------

    'sim_transfer_cube_3d_scripted_twoeyes_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_3d_scripted_twoeyes_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'left_wrist', 'right_wrist']
    },

    'sim_transfer_cube_3d_scripted_cross_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_3d_scripted_cross_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'left_wrist', 'right_wrist']
    },

#------------------3D environment: angle + gripper (3 cams) -------------

    'sim_transfer_cube_3d_scripted_angle_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_3d_scripted_angle_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'left_wrist', 'right_wrist']
    },

#------------------3D environment: top + gripper (3 cams) ---------------

    'sim_transfer_cube_3d_scripted_top_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_3d_scripted_top_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

#------------------3D environment: top + angle + gripper (4 cams) -------

    'sim_transfer_cube_3d_scripted_top_angle_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'angle', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_3d_scripted_top_angle_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'angle', 'left_wrist', 'right_wrist']
    },

#------------------3D environment: two eyes vertical (angle+angle_down) -

    'sim_transfer_cube_3d_scripted_twoeyes_vertical':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'angle_down']
    },

    'sim_insertion_3d_scripted_twoeyes_vertical': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'angle_down']
    },

#------------------3D environment: two eyes vertical + gripper ----------

    'sim_transfer_cube_3d_scripted_twoeyes_vertical_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'angle_down', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_3d_scripted_twoeyes_vertical_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'angle_down', 'left_wrist', 'right_wrist']
    },

#------------------3D environment: cross vertical (angle+angle_down) ----

    'sim_transfer_cube_3d_scripted_cross_vertical':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'angle_down']
    },

    'sim_insertion_3d_scripted_cross_vertical': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'angle_down']
    },

#------------------3D environment: cross vertical + gripper -------------

    'sim_transfer_cube_3d_scripted_cross_vertical_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'angle_down', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_3d_scripted_cross_vertical_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'angle_down', 'left_wrist', 'right_wrist']
    },

#------------------3D environment: four eyes (4 cams) -------------------

    'sim_transfer_cube_3d_scripted_foureyes':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'angle', 'angle_down']
    },

    'sim_insertion_3d_scripted_foureyes': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'angle', 'angle_down']
    },

#------------------3D environment: four eyes + gripper (6 cams) ---------

    'sim_transfer_cube_3d_scripted_foureyes_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'angle', 'angle_down', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_3d_scripted_foureyes_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'angle', 'angle_down', 'left_wrist', 'right_wrist']
    },

#------------------3D environment: two eyes + top + gripper (5 cams) ----

    'sim_transfer_cube_3d_scripted_twoeyes_top_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_3d_scripted_twoeyes_top_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'top', 'left_wrist', 'right_wrist']
    },

#------------------3D environment: four eyes cross (4 cams) -------------

    'sim_transfer_cube_3d_scripted_foureyes_cross':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'angle', 'angle_down']
    },

    'sim_insertion_3d_scripted_foureyes_cross': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'angle', 'angle_down']
    },

#------------------3D environment: four eyes cross + gripper (6 cams) ---

    'sim_transfer_cube_3d_scripted_foureyes_cross_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'angle', 'angle_down', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_3d_scripted_foureyes_cross_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_3d_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'angle', 'angle_down', 'left_wrist', 'right_wrist']
    },

#------------------two eyes simulation ------------------------

    'sim_transfer_cube_scripted_twoeyes':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ["left_pillar", "right_pillar"]
    },

    'sim_insertion_scripted_twoeyes': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ["left_pillar", "right_pillar"]
    },

    'sim_transfer_cube_scripted_twoeyes_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ["left_pillar", "right_pillar",'left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_twoeyes_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ["left_pillar", "right_pillar",'left_wrist', 'right_wrist']
    },

#------------------cross eyes simulation ------------------------

    'sim_transfer_cube_scripted_cross':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ["left_pillar", "right_pillar"]
    },

    'sim_insertion_scripted_cross': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ["left_pillar", "right_pillar"]
    },

    'sim_transfer_cube_scripted_cross_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ["left_pillar", "right_pillar",'left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_cross_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ["left_pillar", "right_pillar",'left_wrist', 'right_wrist']
    },

#------------------up and down eyes simulation vertical------------------------

    'sim_transfer_cube_scripted_twoeyes_vertical':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'angle_down']
    },

    'sim_insertion_scripted_twoeyes_vertical': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'angle_down']
    },

    'sim_transfer_cube_scripted_cross_vertical':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'angle_down']
    },

    'sim_insertion_scripted_cross_vertical': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'angle_down']
    },

#------------------up and down eyes + gripper simulation vertical -----------

    'sim_transfer_cube_scripted_twoeyes_vertical_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'angle_down', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_twoeyes_vertical_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'angle_down', 'left_wrist', 'right_wrist']
    },

    'sim_transfer_cube_scripted_cross_vertical_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'angle_down', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_cross_vertical_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'angle_down', 'left_wrist', 'right_wrist']
    },

#------------------gripper only simulation ------------------------

    'sim_transfer_cube_scripted_gripper_only':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_gripper_only': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_wrist', 'right_wrist']
    },

#------------------angle + gripper simulation (3 cams) ------------------------

    'sim_transfer_cube_scripted_angle_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_angle_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['angle', 'left_wrist', 'right_wrist']
    },

#------------------top + gripper simulation (3 cams) ------------------------

    'sim_transfer_cube_scripted_top_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_top_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

#------------------top + angle simulation (2 cams) -------------------------

    'sim_transfer_cube_scripted_top_angle':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'angle']
    },

    'sim_insertion_scripted_top_angle': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'angle']
    },

#------------------top + angle + gripper simulation (4 cams) ---------------

    'sim_transfer_cube_scripted_top_angle_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'angle', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_top_angle_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'angle', 'left_wrist', 'right_wrist']
    },

#------------------four eyes simulation (horizontal + vertical, 4 cams) -----

    'sim_transfer_cube_scripted_foureyes':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'angle', 'angle_down']
    },

    'sim_insertion_scripted_foureyes': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'angle', 'angle_down']
    },

    'sim_transfer_cube_scripted_foureyes_cross':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'angle', 'angle_down']
    },

    'sim_insertion_scripted_foureyes_cross': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'angle', 'angle_down']
    },

#------------------four eyes + gripper simulation (6 cams) -----------------

    'sim_transfer_cube_scripted_foureyes_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'angle', 'angle_down', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_foureyes_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'angle', 'angle_down', 'left_wrist', 'right_wrist']
    },

    'sim_transfer_cube_scripted_foureyes_cross_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'angle', 'angle_down', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_foureyes_cross_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'angle', 'angle_down', 'left_wrist', 'right_wrist']
    },

#------------------twoeyes + top + gripper simulation (5 cams) -------------

    'sim_transfer_cube_scripted_twoeyes_top_gripper':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_twoeyes_top_gripper': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['left_pillar', 'right_pillar', 'top', 'left_wrist', 'right_wrist']
    },

#------------------gym_guided_vision 3-arm tasks ------------------------
# 2-arm gv task config는 등록하지 않음 (2-arm 실험은 기존 sim_env 기반으로 진행)

    'gv_sim_insert_peg_3arms': {
        'dataset_dir': DATA_DIR + '/gv_sim_insert_peg_3arms',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['zed_cam_left', 'zed_cam_right'],
        'state_dim': 21,
        'num_arms': 3,
    },

    'gv_sim_slot_insertion_3arms': {
        'dataset_dir': DATA_DIR + '/gv_sim_slot_insertion_3arms',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['zed_cam_left', 'zed_cam_right'],
        'state_dim': 21,
        'num_arms': 3,
    },

    'gv_sim_sew_needle_3arms': {
        'dataset_dir': DATA_DIR + '/gv_sim_sew_needle_3arms',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['zed_cam_left', 'zed_cam_right'],
        'state_dim': 21,
        'num_arms': 3,
    },

    'gv_sim_tube_transfer_3arms': {
        'dataset_dir': DATA_DIR + '/gv_sim_tube_transfer_3arms',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['zed_cam_left', 'zed_cam_right'],
        'state_dim': 21,
        'num_arms': 3,
    },

    'gv_sim_hook_package_3arms': {
        'dataset_dir': DATA_DIR + '/gv_sim_hook_package_3arms',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['zed_cam_left', 'zed_cam_right'],
        'state_dim': 21,
        'num_arms': 3,
    },

#------------------av_aloha (new) 3-arm stereo (zed L/R) tasks ------
# 데이터: HuggingFace iantc104/av_aloha_sim_* → /home/kist/data/aloha/av_aloha_sim_*
# 환경:  gym_av_aloha (VITA, ICLR-2026) — env/gv_sim_env.py가 task 이름으로 dispatch

    'av_aloha_sim_cube_transfer_twoeyes': {
        'dataset_dir': DATA_DIR + '/av_aloha_sim_cube_transfer',
        'num_episodes': 200, 'episode_len': 400,
        'camera_names': ['zed_cam_left', 'zed_cam_right'],
        'state_dim': 21, 'num_arms': 3,
    },
    'av_aloha_sim_cube_transfer_cross': {
        'dataset_dir': DATA_DIR + '/av_aloha_sim_cube_transfer',
        'num_episodes': 200, 'episode_len': 400,
        'camera_names': ['zed_cam_left', 'zed_cam_right'],
        'state_dim': 21, 'num_arms': 3,
    },
    'av_aloha_sim_thread_needle_twoeyes': {
        'dataset_dir': DATA_DIR + '/av_aloha_sim_thread_needle',
        'num_episodes': 200, 'episode_len': 400,
        'camera_names': ['zed_cam_left', 'zed_cam_right'],
        'state_dim': 21, 'num_arms': 3,
    },
    'av_aloha_sim_thread_needle_cross': {
        'dataset_dir': DATA_DIR + '/av_aloha_sim_thread_needle',
        'num_episodes': 200, 'episode_len': 400,
        'camera_names': ['zed_cam_left', 'zed_cam_right'],
        'state_dim': 21, 'num_arms': 3,
    },
    'av_aloha_sim_pour_test_tube_twoeyes': {
        'dataset_dir': DATA_DIR + '/av_aloha_sim_pour_test_tube',
        'num_episodes': 100, 'episode_len': 400,
        'camera_names': ['zed_cam_left', 'zed_cam_right'],
        'state_dim': 21, 'num_arms': 3,
    },
    'av_aloha_sim_pour_test_tube_cross': {
        'dataset_dir': DATA_DIR + '/av_aloha_sim_pour_test_tube',
        'num_episodes': 100, 'episode_len': 400,
        'camera_names': ['zed_cam_left', 'zed_cam_right'],
        'state_dim': 21, 'num_arms': 3,
    },
    'av_aloha_sim_slot_insertion_twoeyes': {
        'dataset_dir': DATA_DIR + '/av_aloha_sim_slot_insertion',
        'num_episodes': 100, 'episode_len': 400,
        'camera_names': ['zed_cam_left', 'zed_cam_right'],
        'state_dim': 21, 'num_arms': 3,
    },
    'av_aloha_sim_slot_insertion_cross': {
        'dataset_dir': DATA_DIR + '/av_aloha_sim_slot_insertion',
        'num_episodes': 100, 'episode_len': 400,
        'camera_names': ['zed_cam_left', 'zed_cam_right'],
        'state_dim': 21, 'num_arms': 3,
    },
    'av_aloha_sim_hook_package_twoeyes': {
        'dataset_dir': DATA_DIR + '/av_aloha_sim_hook_package',
        'num_episodes': 100, 'episode_len': 400,
        'camera_names': ['zed_cam_left', 'zed_cam_right'],
        'state_dim': 21, 'num_arms': 3,
    },
    'av_aloha_sim_hook_package_cross': {
        'dataset_dir': DATA_DIR + '/av_aloha_sim_hook_package',
        'num_episodes': 100, 'episode_len': 400,
        'camera_names': ['zed_cam_left', 'zed_cam_right'],
        'state_dim': 21, 'num_arms': 3,
    },

#------------------av_aloha (new) 3-arm angle (single zed_cam_left) tasks -----
    'av_aloha_sim_cube_transfer_angle': {
        'dataset_dir': DATA_DIR + '/av_aloha_sim_cube_transfer',
        'num_episodes': 200, 'episode_len': 400,
        'camera_names': ['zed_cam_left'],
        'state_dim': 21, 'num_arms': 3,
    },
    'av_aloha_sim_thread_needle_angle': {
        'dataset_dir': DATA_DIR + '/av_aloha_sim_thread_needle',
        'num_episodes': 200, 'episode_len': 400,
        'camera_names': ['zed_cam_left'],
        'state_dim': 21, 'num_arms': 3,
    },
    'av_aloha_sim_pour_test_tube_angle': {
        'dataset_dir': DATA_DIR + '/av_aloha_sim_pour_test_tube',
        'num_episodes': 100, 'episode_len': 400,
        'camera_names': ['zed_cam_left'],
        'state_dim': 21, 'num_arms': 3,
    },
    'av_aloha_sim_slot_insertion_angle': {
        'dataset_dir': DATA_DIR + '/av_aloha_sim_slot_insertion',
        'num_episodes': 100, 'episode_len': 400,
        'camera_names': ['zed_cam_left'],
        'state_dim': 21, 'num_arms': 3,
    },
    'av_aloha_sim_hook_package_angle': {
        'dataset_dir': DATA_DIR + '/av_aloha_sim_hook_package',
        'num_episodes': 100, 'episode_len': 400,
        'camera_names': ['zed_cam_left'],
        'state_dim': 21, 'num_arms': 3,
    },

#------------------gym_guided_vision 3-arm angle (single camera) tasks ------

    'gv_sim_insert_peg_3arms_angle': {
        'dataset_dir': DATA_DIR + '/gv_sim_insert_peg_3arms',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['zed_cam_left'],
        'state_dim': 21,
        'num_arms': 3,
    },

    'gv_sim_slot_insertion_3arms_angle': {
        'dataset_dir': DATA_DIR + '/gv_sim_slot_insertion_3arms',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['zed_cam_left'],
        'state_dim': 21,
        'num_arms': 3,
    },

    'gv_sim_sew_needle_3arms_angle': {
        'dataset_dir': DATA_DIR + '/gv_sim_sew_needle_3arms',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['zed_cam_left'],
        'state_dim': 21,
        'num_arms': 3,
    },

    'gv_sim_tube_transfer_3arms_angle': {
        'dataset_dir': DATA_DIR + '/gv_sim_tube_transfer_3arms',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['zed_cam_left'],
        'state_dim': 21,
        'num_arms': 3,
    },

    'gv_sim_hook_package_3arms_angle': {
        'dataset_dir': DATA_DIR + '/gv_sim_hook_package_3arms',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['zed_cam_left'],
        'state_dim': 21,
        'num_arms': 3,
    },

#------------------real Franka single arm (8D) ------------------------

    'real_single_franka_angle': {
        'dataset_dir': '',
        'num_episodes': 100,
        'episode_len': None,
        'camera_names': ['left_pillar'],
        'state_dim': 8,
        'num_arms': 1,
    },

    'real_single_franka_twoeyes': {
        'dataset_dir': '',
        'num_episodes': 100,
        'episode_len': None,
        'camera_names': ['left_pillar', 'right_pillar'],
        'state_dim': 8,
        'num_arms': 1,
    },

    'real_single_franka_cross': {
        'dataset_dir': '',
        'num_episodes': 100,
        'episode_len': None,
        'camera_names': ['left_pillar', 'right_pillar'],
        'state_dim': 8,
        'num_arms': 1,
    },

#------------------real Franka single arm (8D) gripper ---------------

    'real_single_franka_angle_gripper': {
        'dataset_dir': '',
        'num_episodes': 100,
        'episode_len': None,
        'camera_names': ['left_pillar', 'left_wrist'],
        'state_dim': 8,
        'num_arms': 1,
    },

    'real_single_franka_twoeyes_gripper': {
        'dataset_dir': '',
        'num_episodes': 100,
        'episode_len': None,
        'camera_names': ['left_pillar', 'right_pillar', 'left_wrist'],
        'state_dim': 8,
        'num_arms': 1,
    },

    'real_single_franka_cross_gripper': {
        'dataset_dir': '',
        'num_episodes': 100,
        'episode_len': None,
        'camera_names': ['left_pillar', 'right_pillar', 'left_wrist'],
        'state_dim': 8,
        'num_arms': 1,
    },

#------------------real Franka dual arm (16D) ------------------------

    'real_dual_franka_angle': {
        'dataset_dir': '',
        'num_episodes': 100,
        'episode_len': None,
        'camera_names': ['left_pillar'],
        'state_dim': 16,
        'num_arms': 4,
    },

    'real_dual_franka_twoeyes': {
        'dataset_dir': '',
        'num_episodes': 100,
        'episode_len': None,
        'camera_names': ['left_pillar', 'right_pillar'],
        'state_dim': 16,
        'num_arms': 4,
    },

    'real_dual_franka_cross': {
        'dataset_dir': '',
        'num_episodes': 100,
        'episode_len': None,
        'camera_names': ['left_pillar', 'right_pillar'],
        'state_dim': 16,
        'num_arms': 4,
    },

#------------------real Franka dual arm (16D) gripper ------------------------

    'real_dual_franka_angle_gripper': {
        'dataset_dir': '',
        'num_episodes': 100,
        'episode_len': None,
        'camera_names': ['left_pillar', 'left_wrist', 'right_wrist'],
        'state_dim': 16,
        'num_arms': 4,
    },

    'real_dual_franka_twoeyes_gripper': {
        'dataset_dir': '',
        'num_episodes': 100,
        'episode_len': None,
        'camera_names': ['left_pillar', 'right_pillar', 'left_wrist', 'right_wrist'],
        'state_dim': 16,
        'num_arms': 4,
    },

    'real_dual_franka_cross_gripper': {
        'dataset_dir': '',
        'num_episodes': 100,
        'episode_len': None,
        'camera_names': ['left_pillar', 'right_pillar', 'left_wrist', 'right_wrist'],
        'state_dim': 16,
        'num_arms': 4,
    },

#------------------real IGRIS-C single arm 9-DoF (right arm 7 + right thumb + right index) ----
# Dataset: /home/kist/igris_teleop/igris_artifacts/datasets_class_fanta/IGRIS_C_20260524_183810 (52 ep, 30Hz)
# state/action layout (9D): [right_arm 7 | right_thumb (hand[0]) | right_index (hand[1])]
# Note: --split_decoder is unsupported for num_arms=5 (detr_vae.py allows only 2/3/4). Do not use.

    'real_single_igris_9d_angle': {
        'dataset_dir': '',
        'num_episodes': 52,
        'episode_len': None,
        'camera_names': ['right_pillar'],
        'state_dim': 9,
        'num_arms': 5,
    },

    'real_single_igris_9d_twoeyes': {
        'dataset_dir': '',
        'num_episodes': 52,
        'episode_len': None,
        'camera_names': ['left_pillar', 'right_pillar'],
        'state_dim': 9,
        'num_arms': 5,
    },

    'real_single_igris_9d_cross': {
        'dataset_dir': '',
        'num_episodes': 52,
        'episode_len': None,
        'camera_names': ['left_pillar', 'right_pillar'],
        'state_dim': 9,
        'num_arms': 5,
    },

#------------------real IGRIS-C single arm 9-DoF + gripper cameras (left_wrist + right_wrist) ----
# 위 angle/twoeyes/cross 변종에 그리퍼(wrist) 카메라를 추가한 버전.
# preprocess_class_fanta.py 가 추출하는 5개 카메라 중 wrist 2종을 포함.


    'real_single_igris_9d_angle_gripper': {
        'dataset_dir': '',
        'num_episodes': 52,
        'episode_len': None,
        'camera_names': ['right_pillar', 'right_wrist'],
        'state_dim': 9,
        'num_arms': 5,
    },

    'real_single_igris_9d_twoeyes_gripper': {
        'dataset_dir': '',
        'num_episodes': 52,
        'episode_len': None,
        'camera_names': ['left_pillar', 'right_pillar', 'right_wrist'],
        'state_dim': 9,
        'num_arms': 5,
    },

    'real_single_igris_9d_cross_gripper': {
        'dataset_dir': '',
        'num_episodes': 52,
        'episode_len': None,
        'camera_names': ['left_pillar', 'right_pillar', 'right_wrist'],
        'state_dim': 9,
        'num_arms': 5,
    },

#------------------cube_classification 학습용 (script_45 cube_classification 연동) -----

    'sim_cube_classification_scripted':{
        'dataset_dir': DATA_DIR + '/sim_cube_classification_scripted',
        'num_episodes': 120,
        'episode_len': 1700,
        'camera_names': ['top']
    },
    'sim_cube_classification_scripted_angle':{
        'dataset_dir': DATA_DIR + '/sim_cube_classification_scripted',
        'num_episodes': 120,
        'episode_len': 1700,
        'camera_names': ['angle']
    },
    'sim_cube_classification_scripted_gripper_only':{
        'dataset_dir': DATA_DIR + '/sim_cube_classification_scripted',
        'num_episodes': 120,
        'episode_len': 1700,
        'camera_names': ['left_wrist', 'right_wrist']
    },
    'sim_cube_classification_scripted_top_angle':{
        'dataset_dir': DATA_DIR + '/sim_cube_classification_scripted',
        'num_episodes': 120,
        'episode_len': 1700,
        'camera_names': ['top', 'angle']
    },
    'sim_cube_classification_scripted_cross':{
        'dataset_dir': DATA_DIR + '/sim_cube_classification_scripted',
        'num_episodes': 120,
        'episode_len': 1700,
        'camera_names': ['left_pillar', 'right_pillar']
    },
    'sim_cube_classification_scripted_twoeyes':{
        'dataset_dir': DATA_DIR + '/sim_cube_classification_scripted',
        'num_episodes': 120,
        'episode_len': 1700,
        'camera_names': ['left_pillar', 'right_pillar']
    },
}

### Simulation envs fixed constants
DT = 0.02
GV_DT = 0.04
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
