#!/bin/bash
#
# Armlab Setup Script - Workspace & ROS2 Dependencies Only
#

echo "🚀 Starting Workspace Setup..."

# 1. INSTALL PYTHON & ROS SYSTEM DEPENDENCIES 
# (Things not in the Dockerfile but needed for your specific ROS nodes)
echo "Installing Python and ROS-specific packages..."
# Update this section in your install.sh
sudo apt-get update && sudo apt-get install -y \
    python3-pyqt5 \
    pyqt5-dev-tools \
    python3-tk \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-message-filters \
    ros-humble-apriltag-ros \
    ros-humble-diagnostic-updater \
    ros-dev-tools \
    && sudo rm -rf /var/lib/apt/lists/*

# Install specific Python libraries via pip
pip install future modern_robotics

# 2. REALSENSE ROS WRAPPER SETUP
# We keep this here because it involves downloading source code to your workspace
echo "Setting up RealSense ROS Wrapper..."
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

if [ ! -d "realsense-ros-4.54.1" ]; then
    echo "Downloading RealSense ROS wrapper..."
    wget -q https://github.com/IntelRealSense/realsense-ros/archive/refs/tags/4.54.1.zip
    unzip -q 4.54.1.zip
    rm 4.54.1.zip
fi

# 3. RESOLVE DEPENDENCIES & BUILD
cd ~/ros2_ws

echo "Resolving workspace dependencies with rosdep..."
rosdep update --include-eol-distros
# Note: We skip-keys=librealsense2 because it's already installed in the Dockerfile
rosdep install -i --from-path src --rosdistro humble --skip-keys=librealsense2 -y

echo "Building the workspace (this may take a few minutes)..."
source /opt/ros/humble/setup.bash
colcon build --symlink-install

# 4. ENV SETUP (Permanent Sourcing)
echo "Finalizing environment..."
grep -qxF "source /opt/ros/humble/setup.bash" ~/.bashrc || echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
grep -qxF "source ~/ros2_ws/install/setup.bash" ~/.bashrc || echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc

echo "---------------------------------------"
echo "✅ INSTALL COMPLETE."
echo "Restart terminal or run: source ~/.bashrc"