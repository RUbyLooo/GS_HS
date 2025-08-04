#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_DIR=$(realpath "$SCRIPT_DIR/scout_mini_ws")
echo "切换到工作空间目录: $WORKSPACE_DIR"
cd "$WORKSPACE_DIR" || exit 1

if [ ! -f devel/setup.bash ]; then
    echo "❌ 找不到 devel/setup.bash，请先运行 catkin_make"
    exit 1
fi
source devel/setup.bash

sudo modprobe gs_usb
sudo ip link set can0 up type can bitrate 500000

roslaunch scout_bringup scout_mini_robot_base.launch

