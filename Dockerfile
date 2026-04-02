# 1. Start with ROS 2 Humble
FROM --platform=linux/amd64 osrf/ros:humble-desktop

# 2. Environment
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install core tools FIRST (including gnupg2 for the keys)
RUN apt-get update && apt-get install -y \
    sudo curl wget unzip git python3-pip \
    python3-colcon-common-extensions lsb-release \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# 4. Add RealSense Repo (The "Skip the Headache" Method)
RUN echo "deb [trusted=yes] https://librealsense.intel.com/Debian/apt-repo jammy main" | \
    tee /etc/apt/sources.list.d/librealsense.list && \
    apt-get update && apt-get install -y \
    librealsense2-utils \
    librealsense2-dev \
    librealsense2-dbg \
    && rm -rf /var/lib/apt/lists/*

# 5. Set up Workspace
WORKDIR /root/armSorter
COPY . /root/armSorter

# 6. Permissions & Rosdep
RUN chmod +x install_scripts/*.sh launch/*.sh && \
    rm -f /etc/ros/rosdep/sources.list.d/20-default.list && \
    rosdep init && rosdep update

WORKDIR /root/armSorter
CMD ["/bin/bash"]