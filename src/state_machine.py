"""!
The state machine that implements the logic.
"""


from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
from kinematics import IK_geometric
import copy
import math 
import cv2
import threading
from kinematics import clamp
import modern_robotics as mr
from detection import find_block
from detection import detect_uniqueColors
from std_msgs.msg import Int8MultiArray
from rclpy.node import Node

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
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

        self.record_waypoints = []
        self.record_gripper  = []
        self.waypoint_played = False

        self.world_pos = np.empty((3,3))
        self.pick_size = -1

        self.thermal_grid = [0] * 64
        self.human_detected = False
        self.paused_state = None
        self.is_human_paused = False
        
        self.inner_indices = [r * 8 + c for r in range(1, 7) for c in range(1, 7)]

        # self.node = rclpy.create_node('state_machine_thermal_sub')
        # self.thermal_sub = self.node.create_subscription(
        #     Int8MultiArray,
        #     '/thermal_binary_grid',
        #     self.thermal_callback,
        #     10
        # )

        # self.ros_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        # self.ros_thread.start()

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and functions as needed.
        """

        # IMPORTANT: This function runs in a loop. If you make a new state, it will be run every iteration.
        #            The function (and the state functions within) will continuously be called until the state changes.
        active_states = {"execute", "play", "pick", "place", "locate", "detect"}

        if self.human_detected and self.current_state in active_states:
            if not self.is_human_paused:
                self.paused_state = self.next_state   # remember where to return
                self.is_human_paused = True
            self.status_message = "⚠ HUMAN DETECTED — ARM PAUSED"
            self.next_state = "idle"                  # hold in idle while blocked
            self.current_state = "human_paused"
            return                                    # skip all other state logic

        elif self.is_human_paused and not self.human_detected:
            # Human has cleared — resume
            self.is_human_paused = False
        
        else:
            if self.paused_state:
                self.next_state = self.paused_state
                self.paused_state = None
            if self.next_state == "initialize_rxarm":
                self.initialize_rxarm()

            if self.next_state == "idle":
                self.idle()

            if self.next_state == "estop":
                self.estop()

            if self.next_state == "execute":
                self.execute()

            if self.next_state == "calibrate":
                self.calibrate()

            if self.next_state == "detect":
                self.detect()

            if self.next_state == "manual":
                self.manual()

            if self.next_state == "record":
                self.record()
            
            if self.next_state == "play":
                self.play()

            if self.next_state == "human":
                self.human()

            if self.next_state == "pick":
                self.pick()

            if self.next_state == "place":
                self.place()
            if self.next_state == "locate":
                self.get_location()
        


    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.status_message = "State: Execute - Executing motion plan"
        self.rxarm.estop = False 
        self.rxarm.enable_torque()

        for item in self.waypoints:
            if (self.rxarm.estop):
                self.next_state = "estop"
                return
            self.rxarm.set_positions(item)
            time.sleep(2)

        self.next_state = "idle"

    
    def record(self):
        self.status_message = "State: Record - Recording waypoints"
        self.current_state = "record"
        if self.waypoint_played:
            self.record_waypoints = []
            self.record_gripper = []
            self.waypoint_played = False
        self.record_waypoints.append(self.rxarm.get_positions())
        self.record_gripper.append(self.rxarm.gripper_state)
        self.next_state = "idle"


    def play(self):
        self.status_message = "State: Play - Executing recorded motion plan"
        self.current_state = "play"
        
        self.rxarm.estop = False
        self.rxarm.enable_torque()
        self.waypoint_played = True
        for idx, point in enumerate(self.record_waypoints):
            gripper_state = self.record_gripper[idx]
            move_time = 2.0
            ac_time = 0.5
            
            if idx > 0:
                pre_point = self.record_waypoints[idx - 1]
                displacement = point - pre_point
                angular_t = np.abs(displacement) / (np.pi / 5)
                move_time = np.max(angular_t)
                ac_time = move_time / 4.0

                self.rxarm.arm.set_joint_positions(point,
                                 moving_time=move_time,
                                 accel_time=ac_time,
                                 blocking=True)
                time.sleep(0.2)   #0.2 seconds work last tested, adjust for lower
            if self.rxarm.estop:
                self.next_state = "estop"
                break
            if gripper_state != self.rxarm.gripper_state:
                if gripper_state:
                    self.rxarm.gripper_release()
                else:
                    self.rxarm.gripper_grasp()
            time.sleep(0.2) 
        if self.rxarm.estop:
            self.next_state = "estop"
        self.next_state = "idle"

    def thermal_callback(self, msg):
        self.thermal_grid = list(msg.data)
        inner_hot = sum(self.thermal_grid[i] for i in self.inner_indices)
        self.human_detected = inner_hot > 10

    def human(self):
        self.status_message = "State: Human - Scanning..."
        self.current_state = "human"

        inner_hot = sum(self.thermal_grid[i] for i in self.inner_indices)

        grid_8x8 = [self.thermal_grid[r*8:(r+1)*8] for r in range(8)]
        inner_only = [[grid_8x8[r][c] for c in range(1, 7)] for r in range(1, 7)]

        if inner_hot > 10:
            self.status_message = f"⚠ HUMAN DETECTED — {inner_hot}/36 inner pixels hot"
            print(f"[HUMAN] Detected: {inner_hot}/36 inner pixels")
            for row in inner_only:
                print(' '.join('█' if cell else '·' for cell in row))
        else:
            self.status_message = f"Human state: Clear ({inner_hot}/36 pixels)"
            print(f"[HUMAN] Clear: {inner_hot}/36 inner pixels")

        self.next_state = "idle"


    def calMoveTime(self, target_joint):
        displacement = target_joint - self.rxarm.get_positions()
        angular_v = np.ones(displacement.shape) * (np.pi / 4)
        angular_v[0] = np.pi / 2.5
        angular_v[3] = np.pi / 2.5
        angular_v[4] = np.pi / 2.5
        angular_t = np.abs(displacement) / angular_v
        move_time = np.max(angular_t)
        if move_time < 0.4:
            move_time = 0.4
        ac_time = move_time / 3
        return move_time, ac_time
    
    # def check_path_clean(self, target_xyz):
    #     blocks = self.camera.block_detections
    #     for i in range(blocks.detected_num):
    #         dist = self.line_dist(blocks.xyzs[i, :2], target_xyz[:2])
    #         # print("path point dist {0}".format(dist))
    #         if dist == 0:
    #             continue
    #         if dist < 50:
    #             return False

    #     return True

    def get_location(self):
        self.camera.new_click = False
        while not self.camera.new_click:
            time.sleep(0.05)

        if self.camera.DepthFrameRaw is None:
            return

        self.camera.new_click = False
        pt = self.camera.last_click

        d = self.camera.DepthFrameRaw[pt[1]][pt[0]]

        color = find_block(self.camera.VideoFrame, pt[0],pt[1])
        print(f"color detected: {color}")
        angle = 0
        center_x = None
        center_y = None
        if color is None:
            angle = 0
        else:
            objects = detect_uniqueColors(self.camera.VideoFrame, color, self.camera, pt[0], pt[1])
            if len(objects) > 0:
                object = objects[0]
                angle = object['angle']
                center_x = object['center_x']
                center_y = object['center_y']
       #     _,_, angle, center_x, center_y = detect_uniqueColors(self.camera.VideoFrame, color, pt[0], pt[1])


        self.camera.pixel_to_World(pt[0], pt[1],d, True) # more accurate

        if len(objects) > 0:
            print(f"angle of block: {angle}")

            print(f"pixelX: {pt[0]}, pixelY: {pt[1]}, depth: {d}")
        #   self.rxarm.arm.get_joint_positions()
        
            print(f"centerX: {center_x}, centerY: {center_y}")
            self.camera.pixel_to_World(int(center_x), int(center_y),d, True) # more accurate




        
        #self.camera.coord_pixel_to_world(pt[0], pt[1], d)
        # if self.rxarm.estop:
        #     self.next_state = "estop"

       # self.camera.blockDetector()

        # canvas = self.camera.drawAlignmentComparison()
        # cv2.imshow("Sensor Alignment Check", canvas)
        # cv2.waitKey(1)
        self.next_state = "idle"




    def pick(self):
        self.camera.new_click = False
        print("[CLICK PICK] Please click one point to pick...")
        while not self.camera.new_click:
            time.sleep(0.05)
        
        self.camera.new_click = False
        pt = self.camera.last_click
        if self.camera.DepthFrameRaw is None:
            return
        z = self.camera.DepthFrameRaw[pt[1]][pt[0]]


        click_uvd = np.append(pt, z)
        target_world_pos, block_ori = self.get_block_xyz_from_click(click_uvd)
        if target_world_pos is None: 
            return 
        
        print("find block color")
        color = find_block(self.camera.VideoFrame, pt[0], pt[1])
        if color is None:
            angle =0
        else:
            print("detect block")
            objects = detect_uniqueColors(self.camera.VideoFrame, color, None,  pt[0], pt[1])
            object = objects[0]
            angle = object['angle']
            center_x = object['center_x']
            center_y = object['center_y']
          #  _,_, angle, center_x, center_y = detect_uniqueColors(self.camera.VideoFrame, color, pt[0], pt[1])
            if angle is None:
                print("angle of block not found")
                angle = 0

           # target_world_pos, block_ori = self.get_block_xyz_from_click(click_uvd)
            center_pos = self.camera.pixel_to_World(center_x,center_y,z, True)

            if (center_pos[0] > 10):

                target_world_pos[0] = center_pos[0] -10
                target_world_pos[1] = center_pos[1] +10
            else :
                target_world_pos[0] = center_pos[0] + 10
            print(f"center {center_pos[0]} {center_pos[1]-10}")

            if (center_pos[1] < 0):
                target_world_pos[1] +=10

        rad = math.radians(angle)
        self.rxarm.arm.go_to_home_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)
        self.status_message = "State: Pick - Click to pick"
        self.current_state = "pick"
        self.auto_pick1(target_world_pos, block_ori, rad)
        # if self.rxarm.estop:
        #     self.next_state = "estop"
        self.next_state = "idle"



    def auto_pick1(self, _target_world_pos, block_ori, angle, phi=np.pi/2, double_check=False, to_sky=False):

        
        target_world_pos = copy.deepcopy(_target_world_pos)
        above_world_pos = copy.deepcopy(_target_world_pos)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!! pick pos:", target_world_pos)
        print("angle of object:" , angle)
        xy_norm = np.linalg.norm(target_world_pos[:2])
        print(xy_norm)
        if xy_norm >= 315 and xy_norm<=430:
            target_world_pos[2] = target_world_pos[2] + 1/55 * xy_norm
            print(1/50 * xy_norm)

        if (target_world_pos[2] < 10): # height too low
            print("selected oject height too close to ground, auto-pick exits")
            return
        
        ############ Planning #############
        # print("[PICK] Planning waypoints...")
        pick_stable = True

        # determine if final pose is reachable, exit if not
        current_joints = self.rxarm.arm.get_joint_commands()


        
        joint_angles_2, reachable_low = self.rxarm.arm.set_ee_pose_components(x=target_world_pos[1]/1000, # (x,y) plane of robot and world frame is rotated
                                                                            y=-target_world_pos[0]/1000, # motor's x axis is flipped
                                                                            z=((target_world_pos[2]/1000)), # position converted to meters
                                                                            pitch = phi,
                                                                            moving_time = 4,
                                                                            execute = False)
        
        pick_height_offset = 0
        if target_world_pos[2] > 100:
            pick_height_offset = 40  # smaller height offset in case the goal becomes unreachable
        else:
            pick_height_offset = 100    

        target_world_pos[2] = target_world_pos[2]  + pick_height_offset



        
        
        if not reachable_low:
            print("final EE pose is not reachable")
            return





        # Define how long the move takes and how many waypoints to check
        time_in_seconds = 4.0
        number_of_waypoints = 100 # Higher number = finer resolution for obstacle checking

        #  generates a path to final destination
        trajectory = mr.JointTrajectory(current_joints, joint_angles_2, time_in_seconds, number_of_waypoints, 3)

        

       # reachable_low, reachable_high = False, False

        # Try vertical reach with phi = pi/2
        # this function computes the path to get to the final EE and execute it if its valid
        # joint_angles_2, reachable_low = self.rxarm.arm.set_ee_pose_components(x=target_world_pos[1]/1000, # (x,y) plane of robot and world frame is rotated
        #                                                                       y=-target_world_pos[0]/1000, # motor's x axis is flipped
        #                                                                       z=(target_world_pos[2])/1000, # position converted to meters
        #                                                                       pitch = phi,
        #                                                                       moving_time =2,
        #                                                                       blocking= True,
        #                                                                       execute = True)
    
        for angles in trajectory:
            self.rxarm.set_positions(angles)
            

        # TODO: inside this function, check if there's obstacles. if so, don't execute the motor command
        # success = self.rxarm.arm.set_ee_cartesian_trajectory(x=target_world_pos[1]/1000, # (x,y) plane of robot and world frame is rotated
        #                                             y=-target_world_pos[0]/1000, # motor's x axis is flipped
        #                                             z=(target_world_pos[2])/1000, # position converted to meters
        #                                             pitch = phi,
        #                                             moving_time =2)

        # if success is False:
        #     print("path to goal failed in autopick")
        #     return
        

        # EE descends to grab the object
        descend_offset = (-pick_height_offset)/1000  
        print("EE descending")        

        # TODO: orient the wrist to align with object orientation before grasping
        joint_angles_2, valid = self.rxarm.arm.set_ee_pose_components(x=target_world_pos[1]/1000, # (x,y) plane of robot and world frame is rotated
                                                                            y=-target_world_pos[0]/1000, # motor's x axis is flipped
                                                                            z=((target_world_pos[2]/1000)), # position converted to meters
                                                                            pitch = phi,
                                                                            moving_time = 4,
                                                                            execute = False)
        if valid is False:
            print("failed to descend arm")
            return
        
        print(f"joint angles from path: {joint_angles_2}")
     
        joint_angles_2[-1] = joint_angles_2[0] + angle # parallel to x axis
        print(f"joint angle base: {joint_angles_2[0]}")



        # orient the wrist before descend
        self.rxarm.arm._publish_commands(joint_angles_2, moving_time=2, accel_time=0, blocking=True)


        #TODO: check for obstacle before descending
        # maintain the orientation while descending

        
        joint_angles_2, reachable_low = self.rxarm.arm.set_ee_pose_components(x=target_world_pos[1]/1000, # (x,y) plane of robot and world frame is rotated
                                                                              y=-target_world_pos[0]/1000, # motor's x axis is flipped
                                                                              z=((target_world_pos[2]/1000)+descend_offset), # position converted to meters
                                                                              pitch = phi,
                                                                              roll = joint_angles_2[-1],
                                                                              moving_time = 4,
                                                                              execute = True)

        self.rxarm.gripper_grasp()
        return
            
    def auto_pick(self, _target_world_pos, block_ori, angle, phi=np.pi/2, double_check=False, to_sky=False):

        
        target_world_pos = copy.deepcopy(_target_world_pos)
        above_world_pos = copy.deepcopy(_target_world_pos)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!! pick pos:", target_world_pos)
        print("angle of object:" , angle)
        xy_norm = np.linalg.norm(target_world_pos[:2])
        print(xy_norm)
        if xy_norm >= 315 and xy_norm<=430:
            target_world_pos[2] = target_world_pos[2] + 1/55 * xy_norm
            print(1/50 * xy_norm)

        if (target_world_pos[2] < 10): # height too low
            print("selected oject height too close to ground, auto-pick exits")
            return False
        
        ############ Planning #############
        # print("[PICK] Planning waypoints...")
        pick_stable = True

        # determine if final pose is reachable, exit if not
        current_joints = self.rxarm.arm.get_joint_commands()


        
        joint_angles_2, reachable_low = self.rxarm.arm.set_ee_pose_components(x=target_world_pos[1]/1000, # (x,y) plane of robot and world frame is rotated
                                                                            y=-target_world_pos[0]/1000, # motor's x axis is flipped
                                                                            z=((target_world_pos[2]/1000)), # position converted to meters
                                                                            pitch = phi,
                                                                            moving_time = 4,
                                                                            execute = False)
        
        pick_height_offset = 0
        if target_world_pos[2] > 100:
            pick_height_offset = 40  # smaller height offset in case the goal becomes unreachable
        else:
            pick_height_offset = 100    

        target_world_pos[2] = target_world_pos[2]  + pick_height_offset


        
        if not reachable_low:
            print("final EE pose is not reachable")
            return False




        # Define how long the move takes and how many waypoints to check
        # time_in_seconds = 4.0
        # number_of_waypoints = 100 # Higher number = finer resolution for obstacle checking

        #  generates a path to final destination
      #  trajectory = mr.JointTrajectory(current_joints, joint_angles_2, time_in_seconds, number_of_waypoints, 3)


       # reachable_low, reachable_high = False, False

        # Try vertical reach with phi = pi/2
        # this function computes the path to get to the final EE and execute it if its valid
        joint_angles_2, reachable_low = self.rxarm.arm.set_ee_pose_components(x=target_world_pos[1]/1000, # (x,y) plane of robot and world frame is rotated
                                                                              y=-target_world_pos[0]/1000, # motor's x axis is flipped
                                                                              z=(target_world_pos[2])/1000, # position converted to meters
                                                                              pitch = phi,
                                                                              moving_time =2,
                                                                              blocking= True,
                                                                              execute = True)
    
        # for angles in trajectory:
        #     self.rxarm.set_positions(angles)
            

        # TODO: inside this function, check if there's obstacles. if so, don't execute the motor command
        # success = self.rxarm.arm.set_ee_cartesian_trajectory(x=target_world_pos[1]/1000, # (x,y) plane of robot and world frame is rotated
        #                                             y=-target_world_pos[0]/1000, # motor's x axis is flipped
        #                                             z=(target_world_pos[2])/1000, # position converted to meters
        #                                             pitch = phi,
        #                                             moving_time =2)

        # if success is False:
        #     print("path to goal failed in autopick")
        #     return
        

        # EE descends to grab the object
        descend_offset = (-pick_height_offset)/1000  
        print("EE descending")        

        # TODO: orient the wrist to align with object orientation before grasping

        EE_angle = (joint_angles_2[0] + angle) 
        if EE_angle >= 0:
            EE_angle = EE_angle % (np.pi / 2)
        else:
            EE_angle = - (abs(EE_angle) % (np.pi/2))
        joint_angles_2, valid = self.rxarm.arm.set_ee_pose_components(x=target_world_pos[1]/1000, # (x,y) plane of robot and world frame is rotated
                                                                            y=-target_world_pos[0]/1000, # motor's x axis is flipped
                                                                            z=((target_world_pos[2]/1000)), # position converted to meters
                                                                            pitch = phi,
                                                                            roll = EE_angle,
                                                                         #   moving_time = 2,
                                                                            execute = True)
        if valid is False:
            print("failed to descend arm")
            return False
        
        print(f"joint angles from path: {joint_angles_2}")
     
        joint_angles_2[-1] = joint_angles_2[0] + angle # parallel to x axis
        print(f"joint angle base: {joint_angles_2[0]}")



        # orient the wrist before descend
       # self.rxarm.arm._publish_commands(joint_angles_2, moving_time=2, accel_time=0, blocking=True)


        #TODO: check for obstacle before descending
        # maintain the orientation while descending

        
        joint_angles_2, reachable_low = self.rxarm.arm.set_ee_pose_components(x=target_world_pos[1]/1000, # (x,y) plane of robot and world frame is rotated
                                                                              y=-target_world_pos[0]/1000, # motor's x axis is flipped
                                                                              z=((target_world_pos[2]/1000)+descend_offset), # position converted to meters
                                                                              pitch = phi,
                                                                              roll = EE_angle,
                                                                              execute = True)

        self.rxarm.gripper_grasp()

        height = (((target_world_pos[2]+100)/1000)+descend_offset)
        joint_angles_2, reachable_low = self.rxarm.arm.set_ee_pose_components(x=target_world_pos[1]/1000, # (x,y) plane of robot and world frame is rotated
                                                                              y=-target_world_pos[0]/1000, # motor's x axis is flipped
                                                                              z=height, # position converted to meters
                                                                              pitch = phi,
                                                                              roll = EE_angle,
                                                                              execute = True)


        return True
            
     

    def place(self):
        self.status_message = "State: Place - Click to place"
        self.current_state = "place"
        self.camera.new_click = False
        print("[CLICK PLACE]    Please click one point to pick...")
        while not self.camera.new_click:
            time.sleep(0.1)
        
        self.camera.new_click = False
        pt = self.camera.last_click
        if self.camera.DepthFrameRaw is None:
            return
        z = self.camera.DepthFrameRaw[pt[1]][pt[0]]
        click_uvd = np.append(pt, z)
        target_world_pos, block_ori = self.get_block_xyz_from_click(click_uvd)
        if (target_world_pos is None):
            return

        self.auto_place1(target_world_pos, block_ori)

        # if self.rxarm.estop:
        #     self.next_state = "estop"
        self.next_state = "idle"


    def auto_place1(self, _target_world_pos, block_ori=None, phi=np.pi/2, place_near=False, to_sky=False):

        target_world_pos = copy.deepcopy(_target_world_pos)
        above_world_pos = copy.deepcopy(_target_world_pos)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!! pick pos:", target_world_pos)

        
        ############ Planning #############
        # print("[Place] Planning waypoints...")
        height_offset = 200     # add height to avoid hitting other obstacles along the path
        target_world_pos[2] = target_world_pos[2]  + height_offset

        # EE descends to grab the object
        drop_height_offset = 50     # arm will drop the block with respect of this height from the chosen bin
        descend_offset = (-height_offset+drop_height_offset)/1000  
        print("EE descending")
        joint_angles_2, reachable_low = self.rxarm.arm.set_ee_pose_components(x=target_world_pos[1]/1000, # (x,y) plane of robot and world frame is rotated
                                                                              y=-target_world_pos[0]/1000, # motor's x axis is flipped
                                                                              z=(target_world_pos[2]/1000)+descend_offset, # position converted to meters
                                                                              pitch = phi,
                                                                              moving_time = 3,
                                                                              execute = True)

        # Try vertical reach with phi = pi/2
        # this function computes the path to get to the final EE and execute it if its valid
        joint_angles_2, reachable_low = self.rxarm.arm.set_ee_pose_components(x=target_world_pos[1]/1000, # (x,y) plane of robot and world frame is rotated
                                                                              y=-target_world_pos[0]/1000, # motor's x axis is flipped
                                                                              z=target_world_pos[2]/1000, # position converted to meters
                                                                              pitch = phi,
                                                                              execute = True)

         

        self.rxarm.gripper_release()
        return

    def auto_place(self, _target_world_pos, block_ori=None, phi=np.pi/2, place_near=False, to_sky=False):

        target_world_pos = copy.deepcopy(_target_world_pos)
        above_world_pos = copy.deepcopy(_target_world_pos)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!! drop pos:", target_world_pos)

        
        ############ Planning #############
        # print("[Place] Planning waypoints...")
       # height_offset = 150/1000     # add height to avoid hitting other obstacles along the path
       # target_world_pos[2] = target_world_pos[2]  

        # EE descends to grab the object
        #drop_height_offset = 70     # arm will drop the block with respect of this height from the chosen bin
       # descend_offset = (-height_offset+drop_height_offset)/1000  
        print("EE descending")
        joint_angles_2, reachable_low = self.rxarm.arm.set_ee_pose_components(x=target_world_pos[1]/1000, # (x,y) plane of robot and world frame is rotated
                                                                              y=-target_world_pos[0]/1000, # motor's x axis is flipped
                                                                              z=((target_world_pos[2]/1000) +  self.camera.bin_definitions['bin1']['drop_offset'][2]/1000),# position converted to meters
                                                                              pitch = phi,
                                                                            #  moving_time =3,
                                                                              execute = True,
                                                                              blocking = True)

        # Try vertical reach with phi = pi/2
        # this function computes the path to get to the final EE and execute it if its valid
        # joint_angles_2, reachable_low = self.rxarm.arm.set_ee_pose_components(x=target_world_pos[1]/1000, # (x,y) plane of robot and world frame is rotated
        #                                                                       y=-target_world_pos[0]/1000, # motor's x axis is flipped
        #                                                                       z=target_world_pos[2]/1000, # position converted to meters
        #                                                                       pitch = phi,
        #                                                                       execute = True,
        #                                                                       blocking = True)
      

        self.rxarm.gripper_release()
        return
      
     
   


    def get_block_xyz_from_click(self, click_uvd):
        """!
        @brief      Converts the clicked uvd (pixel + depth) to world coordinates.
        """
        u, v, z = click_uvd[0], click_uvd[1], click_uvd[2]
        
        # Use the function we just added to your Camera class!
        #world_pos = self.camera.coord_pixel_to_world(u, v, z)
        world_pos = self.camera.pixel_to_World(u,v,z, True)
        print("here1")
        
        

        # To get the true orientation, you'd cross-reference this click with 
        # self.camera.block_detections. For now, we will default to 0.0 rad.
        block_ori = 0.0 
  
        
        return world_pos, block_ori
    
    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""
        self.status_message = "Calibration - Completed Calibration"


    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        self.current_state = "detect"
        self.status_message = "Detecting blocks"
        self.camera.blockDetector()
        self.next_state = "idle"
        # time.sleep(1)

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        self.next_state = "idle"

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    updateHumanDetected = pyqtSignal(bool)   # ← add this line
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            self.updateHumanDetected.emit(self.sm.human_detected)
            time.sleep(0.05)