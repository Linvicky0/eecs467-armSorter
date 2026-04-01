#!/bin/bash
#
# Install Script for Armlab
#
sudo rm -f /etc/apt/sources.list.d/ros2.list
sudo rm -f /etc/apt/sources.list.d/ros-latest.list
sudo rm -f /usr/share/keyrings/ros-archive-keyring.gpg
#initial update
sudo apt-get update
sudo apt-get upgrade

# dev stuff / code tools
sudo apt-get -y install curl wget build-essential cmake dkms \
    git autoconf automake autotools-dev gdb libglib2.0-dev libgtk2.0-dev \
    libusb-dev libusb-1.0-0-dev freeglut3-dev libboost-dev libgsl-dev \
    net-tools doxygen  

# wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
# sudo add-apt-repository -y "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
# sudo apt-get update
# sudo apt install -y code


# Install Qt5 stuff 
sudo apt-get -y install python3-pyqt5
sudo apt -y install pyqt5-dev-tools

#install some python packages
sudo pip install future
sudo pip install modern_robotics

# Install Intel RealSense SDK 2.0 (version 2.54.2 for L515 support)
# Detect Ubuntu codename
CODENAME=$(lsb_release -cs)

# Add Intel RealSense repository as trusted
# echo "deb [trusted=yes] https://librealsense.intel.com/Debian/apt-repo $CODENAME main" | \
#     sudo tee /etc/apt/sources.list.d/librealsense.list

# # Update package lists
# sudo apt update

# # Install specific version 2.54.2 and related packages
# sudo apt-get -y --allow-downgrades install \
#     librealsense2=2.54.2-* \
#     librealsense2-dkms \
#     librealsense2-utils=2.54.2-* \
#     librealsense2-dev=2.54.2-* \
#     librealsense2-dbg=2.54.2-* \
#     librealsense2-gl=2.54.2-*

# # Prevent these packages from being upgraded automatically
# sudo apt-mark hold \
#     librealsense2 \
#     librealsense2-dkms \
#     librealsense2-utils \
#     librealsense2-dev \
#     librealsense2-dbg \
#     librealsense2-gl

# install ROS2
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

sudo apt update
sudo apt upgrade


sudo apt install ros-humble-desktop
sudo apt install ros-humble-ros-base
sudo apt install ros-dev-tools
source /opt/ros/humble/setup.bash

# sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
#         -o /usr/share/keyrings/ros-archive-keyring.gpg
# echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
#      http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
#      | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
# sudo apt update
# sudo apt upgrade
# sudo apt -y install ros-humble-desktop
# sudo apt -y install ros-dev-tools

# add source setup script 
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

#install apriltags
sudo apt-get install -y ros-humble-apriltag-ros

# Install RealSense ROS2 wrapper from ROS servers
# sudo apt install -y ros-humble-realsense2-*  this no longer work because our hardware is out of dated
# mkdir -p ~/ros2_ws/src
# cd ~/ros2_ws/src/
# wget https://github.com/IntelRealSense/realsense-ros/archive/refs/tags/4.54.1.zip
# unzip 4.54.1.zip
# cd ~/ros2_ws
# sudo apt-get install python3-rosdep -y
# sudo rosdep init # "sudo rosdep init --include-eol-distros" for Eloquent and earlier
# rosdep update # "sudo rosdep update --include-eol-distros" for Eloquent and earlier
# ROS_DISTRO=humble  # set your ROS_DISTRO: iron, humble
# rosdep install -i --from-path src --rosdistro humble --skip-keys=librealsense2 -y
# source /opt/ros/humble/setup.bash
# colcon build
# source install/local_setup.bash
# echo "source ~/ros2_ws/install/local_setup.bash" >> ~/.bashrc