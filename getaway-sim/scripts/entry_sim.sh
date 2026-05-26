#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash

export GZ_IP=127.0.0.1
export GZ_PARTITION=getaway
export GZ_RELAY=127.0.0.1

echo "── Starting Xvfb on :99 ──"
Xvfb :99 -screen 0 1280x720x24 +extension GLX +render -noreset > /tmp/xvfb.log 2>&1 &
XVFB_PID=$!
sleep 2
export DISPLAY=:99
echo "Xvfb PID=$XVFB_PID  DISPLAY=$DISPLAY"
glxinfo -B 2>&1 | head -20 || echo "glxinfo failed"

echo "── Launching Gazebo Harmonic with getaway world ──"
gz sim -r -s -v 4 /sim/gazebo/worlds/getaway.sdf > /tmp/gz.log 2>&1 &
GZ_PID=$!

sleep 15

echo "── DIAG: Gazebo PID=$GZ_PID alive? ──"
if kill -0 $GZ_PID 2>/dev/null; then
  echo "Gazebo alive"
else
  echo "Gazebo DEAD — last 80 lines /tmp/gz.log:"
  tail -n 80 /tmp/gz.log
  exit 1
fi

echo "── DIAG: gz topics list (5s timeout) ──"
timeout 5 gz topic -l 2>&1 || echo "gz topic -l timed out / failed"
echo "── DIAG: last 40 lines Gazebo log ──"
tail -n 40 /tmp/gz.log
echo "── DIAG end ──"

echo "── ros_gz_bridge: camera + depth + ambulance teleop ──"
# Nota: en ros_gz_bridge el topic name es el mismo del lado ROS y GZ.
# `--ros-args -r` SOLO renombra el lado ROS — no remapea el lado GZ.
# Por eso publicamos directamente al topic GZ /model/ambulance/cmd_vel
# y dejamos el mismo nombre del lado ROS (raro como nombre ROS, pero funciona).
ros2 run ros_gz_bridge parameter_bridge \
  /world/getaway/model/iris_with_camera/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image@gz.msgs.Image \
  /world/getaway/model/iris_with_camera/link/camera_link/sensor/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo \
  /world/getaway/model/iris_with_camera/link/camera_link/sensor/depth/depth_image@sensor_msgs/msg/Image@gz.msgs.Image \
  /model/ambulance/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist \
  --ros-args -r /world/getaway/model/iris_with_camera/link/camera_link/sensor/camera/image:=/camera/image \
             -r /world/getaway/model/iris_with_camera/link/camera_link/sensor/camera/camera_info:=/camera/info \
             -r /world/getaway/model/iris_with_camera/link/camera_link/sensor/depth/depth_image:=/camera/depth &
BRIDGE_PID=$!

sleep 8

echo "── DIAG: ROS topics ──"
ros2 topic list 2>&1 || true
echo "── DIAG: ROS hz on /camera/image (3s sample) ──"
timeout 3 ros2 topic hz /camera/image 2>&1 || true
echo "── DIAG end ROS ──"

echo "── Starting ambulance motion (7 m/s en círculo R≈60m, +30s delay; pausa si /tmp/manual_mode existe) ──"
(
  sleep 35
  while true; do
    if [ ! -f /tmp/manual_mode ]; then
      # R = v / ω → 60 = 7 / 0.1167
      gz topic -t /model/ambulance/cmd_vel -m gz.msgs.Twist \
        -p "linear: {x: 7.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.1167}" 2>/dev/null || true
    fi
    sleep 1
  done
) &
AMB_PID=$!

echo "── Launching mission script ──"
cd /sim/scripts
python3 drone_refactor_sim.py

kill $GZ_PID $BRIDGE_PID $AMB_PID $XVFB_PID 2>/dev/null || true
wait
