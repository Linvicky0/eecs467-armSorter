import rclpy
import numpy as np
import time
import sys
import select
import termios
import tty
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

"""
2026-01-22 Shaw:

0. Put this script to home directory, this only works on laptops
    that are already set up with interbotix_ws

1. Run the ROS launch command first: 

ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=rx200

2. Run the test: 

python3 rxarm_test.py

3. If you want to stop, Ctrl + C. The robot will stay still.
    To release it:
    either unplug the power, 
    or run rxarm_test.py again so it returns to initial pose
"""

# --- GLOBALS ---
is_paused = True   
go_sleep = False   
toggle_gripper = False
gripper_open = True
current_moving_time = 2.0  
speed_changed = False      

WAYPOINTS = [
            [-np.pi/2,       -0.5,      -0.3,          0.0,        0.0],
            [0.75*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.5*-np.pi/2,   -0.5,      -0.3,      np.pi/2,        0.0],
            [0.25*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.0,             0.0,       0.0,          0.0,        0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [0.5*np.pi/2,     0.5,       0.3,     -np.pi/3,        0.0],
            [0.75*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [np.pi/2,         0.5,       0.3,     -np.pi/3,        0.0],
            [0.0,             0.0,       0.0,          0.0,        0.0]]

# --- INPUT HANDLING ---
def get_key():
    """Reads a single keypress from stdin without waiting for Enter."""
    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        return sys.stdin.read(1)
    return None

def process_input(key):
    global is_paused, go_sleep, toggle_gripper, current_moving_time, speed_changed
    
    if key == ' ':
        is_paused = not is_paused
        print(f"\n[{'PAUSE' if is_paused else 'RUN'}]")
    elif key.lower() == 'q':
        print("\n[QUIT] Moving to sleep pose...")
        go_sleep = True
    elif key.lower() == 'f':
        current_moving_time = max(1.0, current_moving_time - 0.2)
        speed_changed = True
        print(f"\n[SPEED] Faster: {current_moving_time:.1f}s")
    elif key.lower() == 's':
        current_moving_time = min(4.0, current_moving_time + 0.2)
        speed_changed = True
        print(f"\n[SPEED] Slower: {current_moving_time:.1f}s")
    elif key.lower() == 'g':
        toggle_gripper = True

def check_and_apply_gripper_limits(bot):
    """
    Derived from RXArm.check_gripper_state to prevent SDK stalls.
    Explicitly checks finger positions against limits.
    """
    try:
        with bot.core.js_mutex:
            if bot.core.joint_states is None:
                return
            
            # Use left finger index to check limits
            gripper_pos = bot.core.joint_states.position[bot.gripper.left_finger_index]
            
            if (bot.gripper.gripper_moving):
                if (gripper_pos >= bot.gripper.left_finger_upper_limit or 
                    gripper_pos <= bot.gripper.left_finger_lower_limit):
                    
                    # Force stop the gripper motor
                    bot.core.robot_set_motor_registers('single', 'gripper', 'Goal_Position', gripper_pos)
                    bot.gripper.gripper_moving = False
    except Exception:
        pass

def main():
    global is_paused, go_sleep, toggle_gripper, gripper_open, speed_changed
    old_settings = termios.tcgetattr(sys.stdin)
    
    try:
        tty.setcbreak(sys.stdin.fileno())
        rclpy.init()
        
        # Initialize the robot
        bot = InterbotixManipulatorXS(robot_model="rx200")
        
        bot.arm.go_to_home_pose()
        
        print("\n")
        print("\nREADY. Press SPACE to start/pause \
                \n 'G' to toggle gripper \
                \n 'F' to speed up   \
                \n 'S' to slow down \
                \n 'Q' to sleep.")

        waypoint_idx = 0
        while waypoint_idx < len(WAYPOINTS):
            key = get_key()
            if key: process_input(key)

            # 1. Handle Pause / Initial Start
            if is_paused:
                # Stop the arm where it is
                curr = bot.arm.get_joint_commands()
                bot.arm.set_joint_positions(curr, blocking=False)
                while is_paused and not go_sleep:
                    key = get_key()
                    if key: process_input(key)
                    
                    if toggle_gripper:
                        if gripper_open: 
                            bot.gripper.release()
                        else: 
                            bot.gripper.grasp()
                        gripper_open = not gripper_open
                        toggle_gripper = False
                    
                    check_and_apply_gripper_limits(bot)
                    time.sleep(0.05)

            # 2. Handle Emergency Sleep
            if go_sleep:
                break

            # 3. Move to next Waypoint
            wp = WAYPOINTS[waypoint_idx]
            print(f"Moving to waypoint {waypoint_idx+1}/{len(WAYPOINTS)}...")
            bot.arm.set_joint_positions(wp, moving_time=current_moving_time, blocking=False)
            
            # 4. Monitor movement (Blocking loop with escape hatches)
            start_t = time.time()
            while time.time() - start_t < (current_moving_time + 0.2):
                key = get_key()
                if key: process_input(key)
                
                if is_paused or go_sleep: 
                    break 

                if speed_changed:
                    # Re-issue command with new time. 
                    bot.arm.set_joint_positions(wp, moving_time=current_moving_time, blocking=False)
                    speed_changed = False

                # Allow gripper toggle and limit check during motion
                check_and_apply_gripper_limits(bot)
                if toggle_gripper:
                    if gripper_open: bot.gripper.release()
                    else: bot.gripper.grasp()
                    gripper_open = not gripper_open
                    toggle_gripper = False
                time.sleep(0.05)

            # Only increment waypoint if we weren't interrupted
            if not is_paused and not go_sleep:
                waypoint_idx += 1

        print("Done! Moving to Sleep...")
        bot.arm.go_to_sleep_pose()

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C detected. Holding torque for safety.")
    
    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()