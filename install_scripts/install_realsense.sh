#!/bin/bash
# 1. Create directory
sudo mkdir -p /etc/apt/keyrings

# 2. FETCH THE CORRECT KEY (This replaces the broken Intel URL)
curl -s "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xFB0B24895113F120" | gpg --dearmor | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null

# 3. Ensure transport is installed
sudo apt-get install -y apt-transport-https

# 4. Write the source list (Note: we use 'jammy' specifically for ROS Humble compatibility)
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo jammy main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list

# 5. Update and Install
sudo apt update

# Install DKMS and a specific version of the SDK
sudo apt install -y librealsense2-dkms
sudo apt install -y librealsense2=2.55.1-0~realsense.12474 \
                    librealsense2-udev-rules=2.55.1-0~realsense.12474 \
                    librealsense2-utils=2.55.1-0~realsense.12474 \
                    librealsense2-dev=2.55.1-0~realsense.12474

# 6. Install ROS wrapper
sudo apt install -y ros-humble-realsense2-camera