"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
# expm is a matrix exponential function
from scipy.linalg import expm
from math import sin, cos



def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    H = np.identity(4, dtype=np.float64)
    for idx, t in enumerate(joint_angles):
        a, alpha, d, theta = dh_params[idx]
        if idx == link: break   # CHANGE: breaks when joint index == desired link, matrix computed

        if alpha == -1:
            alpha = t
        elif theta == -1:  # maybe use if instead of elif, unless only one of them is guaranteed to be 0
            theta = t
        A = get_transform_from_dh(a, alpha, d, theta)
        H = np.matmul(H, A)

    pose = get_pose_from_T(H)
    return pose


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """
    # same as PrairieLearn
    Rot1 = np.array([[cos(theta), -sin(theta), 0, 0],
                     [sin(theta), cos(theta), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float64)
    Trans1 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, d],
                       [0, 0, 0, 1]], dtype=np.float64)
    Trans2 = np.array([[1, 0, 0, a],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=np.float64)
    Rot2 = np.array([[1, 0, 0, 0],
                     [0, cos(alpha), -sin(alpha), 0],
                     [0, sin(alpha), cos(alpha), 0],
                     [0, 0, 0, 1]], dtype=np.float64)
    T = np.matmul(np.matmul(np.matmul(Rot1, Trans1), Trans2), Rot2)

    return T


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    pass


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    return [0, 0, 0, 0, 0, 0]


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    pass


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """

    d1 = 104.57                 # from t1 to t2, aka base offset
    l1 = np.sqrt(200*200+50*50) # from t2 to t3, shoulder to elbow shortest distance
    l2 = 200                    # from t3 to t4, elbow to wrist
    l3 = 408.575 - 200 - 50     # from t4 to ee, center of gripper (?)
    t_offset = np.arctan2(50, 200) # offset angle bewteen t3 and t2


    # POTENTIAL TODO: theta1, theta3, and theta4's polarity might need to be reversed to match motor config

    # IK for 5DOF 3-linked arm, with 360 rotating base and end effector
    xc,yc,zc, final_theta = pose[0:3]

    R = np.array([cos(final_theta), -sin(final_theta), 0],
                 [sin(final_theta), cos(final_theta), 0],
                 0, 0, 1)
    
    zc = zc - l3

    # find base orientation
    theta1 = np.arctan2(yc, xc)

    # find wrist position
    l1 = dh_params[1]

    # find the wrist orientation
    r_squared = xc**2 + yc**2
    s_squared = (zc - d1)**2    # d1 is the height of base from shoulder

    # elbow down solution
    theta3 = -np.arccos((r_squared + s_squared - l1*l1 - l2*l2)/(2*l1*l2))
    theta2 = np.arctan2(np.sqrt(s_squared), np.sqrt(r_squared)) - np.arctan2(l2*np.sin(theta3), l1 + l2*np.cos(theta3)) 

    theta3 += np.pi/2 -t_offset
    theta3 = -theta3
    theta2 = np.pi/2 - t_offset - theta2 # offset

    theta4 = final_theta- (theta2 + theta3)

    if theta1 >= np.pi or theta1 <= -np.pi:
        return False, [0, 0, 0, 0, 0]

    if theta2 >= np.deg2rad(113) or theta2 <= -np.deg2rad(108):
        return False, [0, 0, 0, 0, 0]

    if theta3 >= np.deg2rad(93) or theta3 <= -np.deg2rad(108):
        return False, [0, 0, 0, 0, 0]

    if theta4 >= np.deg2rad(123) or theta4 <= -np.deg2rad(100):
        return False, [0, 0, 0, 0, 0]

    return [theta1,theta2,theta3,theta4] # ignore theta5 (end effector's orientation) 




    pass