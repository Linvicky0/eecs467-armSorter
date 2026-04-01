# 1. Start with ROS 2 Humble pre-installed
FROM osrf/ros:humble-desktop

# 2. Set environment to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install core system tools your scripts use
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    wget \
    unzip \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# 4. Set up the workspace in the home directory
WORKDIR /root/armSorter
COPY . /root/armSorter

# 5. Fix permissions for your scripts
# Since we are in /root/armSorter, we can use relative paths
RUN chmod +x install_scripts/*.sh launch/*.sh

# 6. Pre-initialize rosdep
RUN rm -f /etc/ros/rosdep/sources.list.d/20-default.list && rosdep init && rosdep update

# Set the default directory when you log in
WORKDIR /root/armSorter

CMD ["/bin/bash"]