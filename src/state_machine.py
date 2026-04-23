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
from kinematics import clamp
import modern_robotics as mr
from detection import find_block
from detection import detect_uniqueColors


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
        if color is None:
            angle = 0
        else:
            _,_, angle, center_x, center_y = detect_uniqueColors(self.camera.VideoFrame, color, pt[0], pt[1])
        print(f"angle of block: {angle}")

        print(f"pixelX: {pt[0]}, pixelY: {pt[1]}, depth: {d}")
     #   self.rxarm.arm.get_joint_positions()

        print(f"centerX: {center_x}, centerY: {center_y}")


        self.camera.pixel_to_World(pt[0], pt[1],d) # more accurate
        self.camera.pixel_to_World(int(center_x), int(center_y),d) # more accurate

        
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
            _,_, angle, center_x, center_y = detect_uniqueColors(self.camera.VideoFrame, color, pt[0], pt[1])
            if angle is None:
                print("angle of block not found")
                angle = 0

           # target_world_pos, block_ori = self.get_block_xyz_from_click(click_uvd)
            center_pos = self.camera.pixel_to_World(center_x,center_y,z)

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
        self.auto_pick(target_world_pos, block_ori, rad)
        # if self.rxarm.estop:
        #     self.next_state = "estop"
        self.next_state = "idle"


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
        joint_angles_2, valid = self.rxarm.arm.set_ee_pose_components(x=target_world_pos[1]/1000, # (x,y) plane of robot and world frame is rotated
                                                                            y=-target_world_pos[0]/1000, # motor's x axis is flipped
                                                                            z=((target_world_pos[2]/1000)), # position converted to meters
                                                                            pitch = phi,
                                                                            roll = EE_angle,
                                                                         #   moving_time = 2,
                                                                            execute = True)
        if valid is False:
            print("failed to descend arm")
            return
        
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
        return
            
     

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

        self.auto_place(target_world_pos, block_ori)

        # if self.rxarm.estop:
        #     self.next_state = "estop"
        self.next_state = "idle"


    def auto_place(self, _target_world_pos, block_ori=None, phi=np.pi/2, place_near=False, to_sky=False):

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
      
     
    def auto_place_notouch(self, _target_world_pos, block_ori=None, phi=np.pi/2, place_near=False, to_sky=False, push=[0,175,0]):
        target_world_pos = copy.deepcopy(_target_world_pos)
        above_world_pos = copy.deepcopy(_target_world_pos)
        push_pos = copy.deepcopy(push)
        if place_near:
            target_world_pos = [0, 200, 0]
            above_world_pos = [0, 200, 0]

        ############ Planning #############
        # print("[PLACE]  Planning waypoints...")
        place_height_offset = 33
        place_wrist_offset = np.pi/18.0/3.0
        target_world_pos[2] = target_world_pos[2] + place_height_offset
        above_world_pos[2] = above_world_pos[2] + place_height_offset + 80
        push_pos[2] = push_pos[2] + place_height_offset

        reachable_low, reachable_high = False, False

        # Try vertical reach with phi = pi/2
        reachable_low, joint_angles_2 = IK_geometric([target_world_pos[0], 
                                                    target_world_pos[1],
                                                    target_world_pos[2],
                                                    phi])

        if reachable_low:
            # phi = np.pi/2
            while not reachable_high:
                reachable_high, joint_angles_1 = IK_geometric([above_world_pos[0], 
                                                            above_world_pos[1],
                                                            above_world_pos[2],
                                                            phi])
                if reachable_high:
                    break
                if above_world_pos[2] - target_world_pos[2] > 40:
                    above_world_pos[2] = above_world_pos[2] - 10
                elif _target_world_pos[2] < 38*4+10:
                    if 0.98* math.sqrt(above_world_pos[0] * above_world_pos[0] + above_world_pos[1] * above_world_pos[1]) > 158.875:
                        above_world_pos[0] = above_world_pos[0] * 0.98
                        above_world_pos[1] = above_world_pos[1] * 0.98
                    phi = phi - np.pi/18.0
                else:
                    break

                if phi <= 0:
                    break

        # Try horizontal reach with phi = 0.0
        if not reachable_high or not reachable_low:
            target_world_pos = copy.deepcopy(_target_world_pos)
            above_world_pos = copy.deepcopy(_target_world_pos)
            if _target_world_pos[2] >= 38*4+10 and to_sky:
                target_world_pos[1] = target_world_pos[1] - 13
                above_world_pos[1] = above_world_pos[1] - 13
            target_world_pos[2] = target_world_pos[2] + 12
            above_world_pos[2] = above_world_pos[2] + 12 + 80
            reachable_low, joint_angles_2 = IK_geometric([target_world_pos[0], 
                                                        target_world_pos[1],
                                                        target_world_pos[2],
                                                        0.0])

            while not reachable_high:
                reachable_high, joint_angles_1 = IK_geometric([above_world_pos[0], 
                                                            above_world_pos[1],
                                                            above_world_pos[2],
                                                            0.0])
                # if 0.95* math.sqrt(above_world_pos[0] * above_world_pos[0] + above_world_pos[1] * above_world_pos[1]) > 158.875:
                #         above_world_pos[0] = above_world_pos[0] * 0.95
                #         above_world_pos[1] = above_world_pos[1] * 0.95
                if reachable_high:
                    break
                if above_world_pos[2] - target_world_pos[2] > 40:
                    above_world_pos[2] = above_world_pos[2] - 10
                else:
                    break

        # Unreachable
        if not reachable_high or not reachable_low:
            if not self.next_state == "estop":
                self.next_state = "idle"
            print("[PLACE]  Target point is unreachable, remain idle!!!")
            return False

        ############ Executing #############
        # print("[PLACE]  Executing waypoints...")
        # !!! TODO
        # joint_angles_start = self.rxarm.get_positions()
        joint_angles_start = [0, 0, 0, 0, 0]
        if _target_world_pos[2] > 38*4+10:
            joint_angles_start = self.rxarm.safe_pose

        joint_angles_start[0] = joint_angles_1[0]
        # move_time = np.abs(joint_angles_1[0] - joint_angles_start[0]) / (np.pi/3)
        # ac_time = move_time / 4
        if not to_sky:
            move_time, ac_time = self.calMoveTime(joint_angles_start)
            print("move time: ", move_time)
            self.rxarm.arm.set_single_joint_position("waist", joint_angles_1[0], moving_time=move_time, accel_time=ac_time, blocking=True)
        else:
            move_time, ac_time = self.calMoveTime(self.rxarm.safe_pose)
            self.rxarm.arm.set_joint_positions(self.rxarm.safe_pose,
                                            moving_time=move_time,
                                            accel_time=ac_time,
                                            blocking=True)

        # 1. go to point above target pose
        joint_angles_1[-2] = joint_angles_1[-2] + place_wrist_offset
        move_time, ac_time = self.calMoveTime(joint_angles_1)
        self.rxarm.arm.set_joint_positions(joint_angles_1,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)

        # 2. go to target pose 
        joint_angles_2[-2] = joint_angles_2[-2] + place_wrist_offset
        displacement = np.array(joint_angles_2) - np.array(joint_angles_1)
        displacement_unit =  displacement

        current_effort = self.rxarm.get_efforts()
        print("initial: ", current_effort)
        temp_joint = np.array(joint_angles_1)
        for i in range(10):
            # current_effort = self.rxarm.get_efforts()
            # print("initial: ", current_effort)
            displacement_unit = displacement_unit / 2
            temp_joint = temp_joint + displacement_unit
            move_time, ac_time = self.calMoveTime(temp_joint)
            self.rxarm.arm.set_joint_positions(temp_joint.tolist(),
                                            moving_time=move_time,
                                            accel_time=ac_time,
                                            blocking=True)
            time.sleep(0.1)
            if i > 0:
                break
        
        reachable_push, joint_angles_3 = IK_geometric([push_pos[0], 
                                                        push_pos[1],
                                                        push_pos[2],
                                                        phi])
        self.rxarm.arm.set_joint_positions(joint_angles_3,
                                        moving_time=2.0,
                                        accel_time=1.0,
                                        blocking=True)

        self.rxarm.open_gripper()
        self.rxarm.gripper_state = False
        
        move_time, ac_time = self.calMoveTime(joint_angles_1)
        self.rxarm.arm.set_joint_positions(joint_angles_1,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)

        joint_angles_end = copy.copy(joint_angles_1)
        joint_angles_end[1] = -np.pi/6
        joint_angles_end[2] = 0
        joint_angles_end[3] = -np.pi/2
        joint_angles_end[4] = 0
        if _target_world_pos[2] >= 38*4+10:
            joint_angles_end[1] = -np.pi/3

        move_time, ac_time = self.calMoveTime(joint_angles_end)
        self.rxarm.arm.set_joint_positions(joint_angles_end,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)
        # print("[PLACE]  Place finished!")
        return True

    def get_block_xyz_from_click(self, click_uvd):
        """!
        @brief      Converts the clicked uvd (pixel + depth) to world coordinates.
        """
        u, v, z = click_uvd[0], click_uvd[1], click_uvd[2]
        
        # Use the function we just added to your Camera class!
        #world_pos = self.camera.coord_pixel_to_world(u, v, z)
        world_pos = self.camera.pixel_to_World(u,v,z)
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
            time.sleep(0.05)