# armlab-f25

**Table of content**
- [Code structure](#code-structure)
- [How to start](#how-to-start)

## Code structure

### Relevant
You do need to modify **some** of these files.
- [install_scripts](install_scripts)
    - [install_scripts/config](install_scripts/config)
        - `rs_l515_launch.py` - to launch the camera
        - `tags_Standard41h12.yaml` - to define the april tags you used on the board
    - `install_Dependencies.sh` - to install ROS2/All the ROS wrappers/Dependencies
    - `install_Interbotix.sh` - to install arm related stuff
    - `install_LaunchFiles.sh` - to move the files under `/config` to where it should to be 
- [launch](launch) - to store the launch files, details in [here](launch/README.md)
- [src](src) - where you actually write code
    - `camera.py` - Implements the Camera class for the RealSense camera. 
        - Functions to capture and convert frames
        - Functions to load camera calibration data
        - Functions to find and perform 2D transforms
        - Functions to perform world-to-camera and camera-to-world transforms
        - Functions to detect blocks in the depth and RGB frames
    - `control_station.py`
         - This is the main program. It sets up the threads and callback functions. Takes flags for whether to use the product of exponentials (PoX) or Denabit-Hartenberg (DH) table for forward kinematics and an argument for the DH table or PoX configuration. You will upgrade some functions and also implement others according to the comments given in the code.
    - `kinematics.py` - Implements functions for forward and inverse kinematics
    - `rxarm.py` - Implements the RXArm class
        - Feedback from joints
        - Functions to command the joints
        - Functions to get feedback from joints
        - Functions to do FK and IK
        - A run function to update the dynamixiel servos
        - A function to read the RX200 arm config file
    - `state_machine.py` - Implements the StateMachine class
        - The state machine is the heart of the controller
- [config](config)
    - `rx200_dh.csv` - Contains the DH table for the RX200 arm
        - You will need to fill this in
    - `rx200_pox.csv` - Containes the S list and M matrix for the RX200 arm.
        - You will need to fill this in


### Irrelevant
Not need to touch these files.
- [media](media) - where we store media that used for README instructions
- [src/resource](src/resource) - where we store the additional files used in the project

## How to start?
- For mac users, download Docker Desktop, increase memory size to 8GB, and run `build_docker.sh` to create a docker image. Now, run the install scripts in `cd ~/root/install_scripts/`. Upon exiting the container and reentering, run `docker start -i armlab-container`

1. Go to [/install_scripts](install_scripts) and following the `README.md` instructions
2. Go to [/launch](launch) to start the ROS2 nodes with the `.sh` files following the `README.md` instructions

## Running Camera
1. Plug Real Sense camera into any usb port labeled with a SS (Usb 3.0). when you plug the real sense camera in, you have to plug it in quickly or else the computer will treat it as a USB2.0 (not good).
2. If you haven't built befpre, you will have to run this command
`cd ~/ros2_ws`
`colcon build --symlink-install --packages-select realsense2_camera --cmake-args -DCMAKE_BUILD_TYPE=Release`
4. So far I have ot updated the launch files, so running this command will boot up the camera to start publishing the topics
`source ~/ros2_ws/install/setup.bash`
`ros2 launch realsense2_camera rs_launch.py device_type:=d435i align_depth.enable:=true`
5. You should get something like this in the terminal
`[realsense2_camera_node-1] [INFO] [1775329998.089389998] [camera.camera]: Open profile: stream_type: Color(0), Format: RGB8, Width: 1280, Height: 720, FPS: 30`
`[realsense2_camera_node-1] [INFO] [1775329998.091468387] [camera.camera]: RealSense Node Is Up!`
If the fornat is 640 x 40, that means the port is being treated as a USB 2.0, which means that you will have to unplug the camera and plug it in faster, or if that isn't work, potentially unplugging the USB-C connection from the camera and flipping it 180 degrees and plugging in might be a solution (haven't verified)
