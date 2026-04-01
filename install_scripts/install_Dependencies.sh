#!/bin/bash
#
# Armlab Setup Script - Universal Docker/M1 Version
#

# 1. PRE-FLIGHT FIXES (The "Manual" stuff we did)
echo "Preparing environment..."
mkdir -p ~/.gnupg && chmod 700 ~/.gnupg
sudo rm -f /etc/apt/sources.list.d/ros2.list # Fix the "Conflicting Values" error

# 2. INTEL REALSENSE KEYS (The Bulletproof Method)
echo "Fetching Intel RealSense keys..."
sudo mkdir -p /usr/share/keyrings

# This pulls the key directly from the Ubuntu trust servers
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/librealsense.gpg --keyserver keyserver.ubuntu.com --recv-keys FB0B24895113F120

# Create the list file pointing to that key
echo "deb [signed-by=/usr/share/keyrings/librealsense.gpg] https://librealsense.intel.com/Debian/apt-repo jammy main" | sudo tee /etc/apt/sources.list.d/librealsense.list


# 3. CORE INSTALLATION
sudo apt-get update
sudo apt-get -y install curl wget build-essential cmake git net-tools python3-pip

# Install Qt5 and Roboties Libraries
sudo apt-get -y install python3-pyqt5 pyqt5-dev-tools
pip install future modern_robotics

# Dependencies for ROS2 Camera Calibration
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-message-filters \
    python3-opencv \
    python3-tk

# 4. REALSENSE SDK
# Install specific version 2.54.2 for L515 support
sudo apt-get -y --allow-downgrades install librealsense2=2.54.2-* librealsense2-utils=2.54.2-* \
                librealsense2-dev=2.54.2-* librealsense2-dbg=2.54.2-* librealsense2-gl=2.54.2-*
sudo apt-mark hold librealsense2 librealsense2-utils librealsense2-dev

# 5. ROS2 & WORKSPACE SETUP
echo "Setting up ROS2 Workspace and RealSense Wrapper..."

# Install tools and AprilTag ROS package
sudo apt update
sudo apt -y install ros-dev-tools ros-humble-apriltag-ros

# Initialize rosdep only if it hasn't been done yet
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update --include-eol-distros

# Setup Workspace
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src

# Download and unzip the wrapper
if [ ! -d "realsense-ros-4.54.1" ]; then
    wget -q https://github.com/IntelRealSense/realsense-ros/archive/refs/tags/4.54.1.zip
    unzip -q 4.54.1.zip
    # No need to rename; colcon will find it inside the unzipped folder
fi

cd ~/ros2_ws

# Install dependencies for the specific source code we just downloaded
# This is the step that usually fixes "missing package" errors during build
rosdep install -i --from-path src --rosdistro humble --skip-keys=librealsense2 -y

# Build the workspace
# Using --symlink-install is a "pro-move" for robotics; it makes future edits faster
source /opt/ros/humble/setup.bash
colcon build --symlink-install

# Make sourcing permanent in .bashrc (with checks to avoid duplicates)
grep -qxF "source /opt/ros/humble/setup.bash" ~/.bashrc || echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
grep -qxF "source ~/ros2_ws/install/local_setup.bash" ~/.bashrc || echo "source ~/ros2_ws/install/local_setup.bash" >> ~/.bashrc

echo "---------------------------------------"
echo "INSTALL COMPLETE. Restart terminal or run: source ~/.bashrc"