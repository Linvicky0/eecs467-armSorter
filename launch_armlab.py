from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        # 1. RealSense Launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_l515_launch.py')
            ])
        ),

        # 2. AprilTag Node
        ExecuteProcess(
            cmd=['ros2', 'run', 'apriltag_ros', 'apriltag_node', 
                 '--ros-args', '-r', 'image_rect:=/camera/color/image_raw', 
                 '-r', 'camera_info:=/camera/color/camera_info', 
                 '--params-file', os.path.join(get_package_share_directory('apriltag_ros'), 'cfg', 'tags_Standard41h12.yaml')],
            output='screen'
        ),

        # 3. XSArm Control Launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('interbotix_xsarm_control'), 'launch', 'xsarm_control.launch.py')
            ]),
            launch_arguments={'robot_model': 'rx200'}.items()
        )
    ])