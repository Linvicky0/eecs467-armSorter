"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
from kinematics import IK_geometric

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

    def pick(self):
        self.status_message = "State: Pick - Click to pick"
        self.current_state = "pick"
        self.camera.new_click = False
        print("[CLICK PICK] Please click one point to pick...")
        while not self.camera.new_click:
            time.sleep(0.05)
        
        self.camera.new_click = False
        pt = self.camera.last_click
        z = self.camera.DepthFrameRaw[pt[1]][pt[0]]
        click_uvd = np.append(pt, z)
        target_world_pos, block_ori = self.get_block_xyz_from_click(click_uvd)

        self.rxarm.go_to_home_pose(moving_time=2,
                                    accel_time=0.5,
                                    blocking=True)
        
        self.auto_pick(target_world_pos, block_ori)
        if self.rxarm.estop:
            self.next_state = "estop"
        self.next_state = "idle"

    def auto_pick(self, _target_world_pos, block_ori, phi=np.pi/2, double_check=False, to_sky=False):
        # if to_sky:
        #     _target_world_pos = [-345, 0, 0]
        target_world_pos = copy.deepcopy(_target_world_pos)
        above_world_pos = copy.deepcopy(_target_world_pos)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!! pick pos:", target_world_pos)
        xy_norm = np.linalg.norm(target_world_pos[:2])
        print(xy_norm)
        if xy_norm >= 315 and xy_norm<=430:
            target_world_pos[2] = target_world_pos[2] + 1/55 * xy_norm
            print(1/50 * xy_norm)
        ############ Planning #############
        # print("[PICK] Planning waypoints...")
        pick_stable = True
        pick_height_offset = 10 + 19 # + 5
        pick_wrist_offset = 0 # np.pi/18.0/3.0
        target_world_pos[2] = target_world_pos[2] + pick_height_offset
        above_world_pos[2] = above_world_pos[2] + pick_height_offset + 80

        reachable_low, reachable_high = False, False

        # Try vertical reach with phi = pi/2
        reachable_low, joint_angles_2 = IK_geometric([target_world_pos[0], 
                                                    target_world_pos[1],
                                                    target_world_pos[2],
                                                    phi])
        # phi = np.pi/2
        if reachable_low:
            while not reachable_high:
                reachable_high, joint_angles_1 = IK_geometric([above_world_pos[0], 
                                                            above_world_pos[1],
                                                            above_world_pos[2],
                                                            phi])
                if reachable_high:
                    break
                if above_world_pos[2] - target_world_pos[2] > 40:
                    above_world_pos[2] = above_world_pos[2] - 10
                else:
                    if 0.98* math.sqrt(above_world_pos[0] * above_world_pos[0] + above_world_pos[1] * above_world_pos[1]) > 158.875:
                        above_world_pos[0] = above_world_pos[0] * 0.98
                        above_world_pos[1] = above_world_pos[1] * 0.98
                    phi = phi - np.pi/18.0
                if phi <= 0:
                    break

        # add horizontal reach by detecting distance between the projection of arm and the target point
        if self.check_path_clean(target_world_pos):
            # Try horizontal reach with phi = 0.0
            target_world_pos = copy.deepcopy(_target_world_pos)
            above_world_pos = copy.deepcopy(_target_world_pos)
            target_world_pos[2] = target_world_pos[2] + 5 + 19
            target_world_pos[0] = target_world_pos[0] * 0.97
            target_world_pos[1] = target_world_pos[1] * 0.97

            above_world_pos[2] = above_world_pos[2] + 5 + 80
            above_world_pos[0] = above_world_pos[0] * 0.97
            above_world_pos[1] = above_world_pos[1] * 0.97
            if not reachable_high or not reachable_low:
                pick_stable = False
                double_check = False
                reachable_low, joint_angles_2 = IK_geometric([target_world_pos[0], 
                                                            target_world_pos[1], 
                                                            target_world_pos[2], 
                                                            0.0])

                reachable_high, joint_angles_1 = IK_geometric([above_world_pos[0], 
                                                            above_world_pos[1], 
                                                            above_world_pos[2], 
                                                            0.0])

        if not reachable_high or not reachable_low:
            if not self.next_state == "estop":
                self.next_state = "idle"
            print("[PICK] Target point is unreachable, remain idle!!!")
            return False, pick_stable

        print("pick high: ", above_world_pos, ' ', phi)
        print("pick low: ", target_world_pos)
        
        ############ Executing #############
        # print("[PICK] Executing waypoints...")
        joint_angles_start = [0, 0, 0, 0, 0]
        joint_angles_start[0] = joint_angles_1[0]

        move_time, ac_time = self.calMoveTime(joint_angles_start)
        self.rxarm.set_single_joint_position("waist", joint_angles_1[0], moving_time=move_time, accel_time=ac_time, blocking=True)

        if to_sky:
            front_world_pos = deepcopy(above_world_pos)
            front_world_pos[0] = front_world_pos[0] + 20
            reachable_front, joint_angles_front = IK_geometric([front_world_pos[0], 
                                                        front_world_pos[1], 
                                                        front_world_pos[2], 
                                                        0.0],
                                                        m_matrix=self.rxarm.M_matrix,
                                                        s_list=self.rxarm.S_list)
            move_time, ac_time = self.calMoveTime(joint_angles_front)
            self.rxarm.set_joint_positions(joint_angles_front,
                                            moving_time=move_time,
                                            accel_time=ac_time,
                                            blocking=True)
        
        joint_angles_1[-2] = joint_angles_1[-2] + pick_wrist_offset
        move_time, ac_time = self.calMoveTime(joint_angles_1)
        self.rxarm.set_joint_positions(joint_angles_1,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)

        # 3. go to the target pose and close gripper
        joint_angles_2[-2] = joint_angles_2[-2] + pick_wrist_offset
        move_time, ac_time = self.calMoveTime(joint_angles_2)
        self.rxarm.set_joint_positions(joint_angles_2,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)
        self.rxarm.close_gripper()
        self.rxarm.gripper_state = False

        if to_sky:
            return True, pick_stable

        if double_check:
            self.rxarm.open_gripper()
            self.rxarm.gripper_state = True
            self.rxarm.set_ee_cartesian_trajectory(z=0.04, moving_time=0.5, wp_moving_time=0.1)
            self.rxarm.set_single_joint_position("wrist_rotate", joint_angles_2[-1] + np.pi/2, moving_time=0.3, accel_time=0.14)
            joint_angles_2[-1] = joint_angles_2[-1] + np.pi/2
            self.rxarm.set_joint_positions(joint_angles_2,
                                            moving_time=move_time,
                                            accel_time=ac_time,
                                            blocking=True)
            self.rxarm.close_gripper()
            self.rxarm.gripper_state = False

        
        # 4. raise to the point above the target point
        move_time, ac_time = self.calMoveTime(joint_angles_1)
        self.rxarm.set_joint_positions(joint_angles_1,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)

        # 5. raise to the theta1 = 0 and theta2 = 0
        joint_angles_end = copy(joint_angles_1)
        joint_angles_end[1] = -np.pi/4
        joint_angles_end[2] = 0
        joint_angles_end[3] = -np.pi/2
        joint_angles_end[4] = 0
        move_time, ac_time = self.calMoveTime(joint_angles_end)
        self.rxarm.set_joint_positions(joint_angles_end,
                                        moving_time=move_time,
                                        accel_time=ac_time,
                                        blocking=True)

        # linear distance between the gripper fingers [m]
        gripper_distance = self.rxarm.get_gripper_position()
        print("!!!!!!!!!!!!!!!!!!!!!!!!! Gripper Dist: {:.8f}".format(gripper_distance))
        # TODO return pick fail according to gripper distance
        # if gripper_distance <= 0.0300000:
        #     print("[PICK] Failed to grab the block!")
        #     return False, pick_stable
        # else:
        # print("[PICK] Pick finished!")
        if gripper_distance>0.04:
            self.pick_size = 0 # large
        else:
            self.pick_size = 1 # small
        
        return True, pick_stable

    def place(self):
        self.status_message = "State: Place - Click to place"
        self.current_state = "place"
        self.camera.new_click = False
        print("[CLICK PLACE]    Please click one point to pick...")
        while not self.camera.new_click:
            time.sleep(0.1)
        
        self.camera.new_click = False
        pt = self.camera.last_click
        z = self.camera.DepthFrameRaw[pt[1]][pt[0]]
        click_uvd = np.append(pt, z)
        target_world_pos, block_ori = self.get_block_xyz_from_click(click_uvd)

        self.auto_place(target_world_pos, block_ori)

        if self.rxarm.estop:
            self.next_state = "estop"
        self.next_state = "idle"

    def get_block_xyz_from_click(self, click_uvd):
        """!
        @brief      Converts the clicked uvd (pixel + depth) to world coordinates.
        """
        u, v, z = click_uvd[0], click_uvd[1], click_uvd[2]
        
        # Use the function we just added to your Camera class!
        world_pos = self.camera.coord_pixel_to_world(u, v, z)
        
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