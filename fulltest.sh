#!/bin/bash
source /home/sgiovanardi/dynamic_reorient_ws/install/setup.bash || true

LOGFILE=/home/sgiovanardi/dynamic_reorient_ws/launch_out.txt

ros2 launch dynamic_reorient gazebo.launch.py > "$LOGFILE" 2>&1 &
LPID=$!

sleep 25

echo "---KEY LINES---"
grep -E "ros2_control|Exception|controller_manager|connected|Received|Error|update_rate|Loaded" "$LOGFILE" | head -20

echo "---COUNTS---"
echo "Exception: $(grep -c 'Exception' "$LOGFILE")"
echo "Loading ros2_control: $(grep -c 'Loading gazebo_ros2_control' "$LOGFILE")"
echo "connected to service: $(grep -c 'connected to service' "$LOGFILE")"
echo "Received urdf: $(grep -c 'Received urdf' "$LOGFILE")"

kill $LPID 2>/dev/null
pkill -9 -f gzserver 2>/dev/null
pkill -9 -f gzclient 2>/dev/null
