#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError

import yaml
DTYPE = np.float64



class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)


        # mouse clicks & calibration variables
        self.camera_calibrated = False
        self.intrinsic_matrix = None
        self.extrinsic_matrix = None
        self.dist_coeff = None
        self.last_click = np.array([0, 0]) # This contains the last clicked position
        self.new_click = False # This is automatically set to True whenever a click is received. Set it to False yourself after processing a click
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275], [-250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])

        # April tag IDS and positions for building the board
        self.boardTag_center =  {  
            4:          # top-left
            3:          # top-right
            1:          # bottom-left
            2:          # bottom-right
        }   

        # load the calibration data
        calibration_file = "calibration_data/ost.yaml"
        self.loadCameraCalibration(calibration_file)

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        if file is not None:
            data = None
            with open(file, "r") as stream:
                data = yaml.safe_load(stream)
            assert (data is not None)
            self.intrinsic_matrix = np.asarray(data["camera_matrix"]["data"], dtype=DTYPE).reshape((3, 3))
            self.dist_coeff = np.asarray(data["distortion_coefficients"]["data"], dtype=DTYPE).reshape(-1)

        else:
            self.intrinsic_matrix = np.array([925.27515, 0.0, 653.75928, 
                                            0.0, 938.70001, 367.99236, 
                                            0.0, 0.0, 1.0], dtype=DTYPE).reshape((3, 3))
        # f22 extrinsic matrix perimeters 
        # self.extrinsic_matrix_inv = np.array([1,0,0,-20,
        #                                     0, -1, 0, 211,
        #                                     0, 0, -1, 974,
        #                                     0, 0, 0, 1], dtype=DTYPE).reshape((4, 4))
        # self.extrinsic_matrix = np.linalg.pinv(self.extrinsic_matrix_inv)

        self.intrinsic_matrix_inv = np.linalg.pinv(self.intrinsic_matrix)


    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        gray = cv2.cvtColor(self.VideoFrame, cv2.COLOR_RGB2GRAY)
        
        # 2. Threshold the image to create a binary mask (black and white)
        # Note: You may need to adjust '127' based on your actual lighting conditions
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 3. Find the contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 4. Store them in the class attribute
        self.block_contours = contours
        
        # 5. Draw the contours onto the VideoFrame
        # -1 draws all contours, (255, 0, 255) is magenta, 3 is thickness
        cv2.drawContours(self.VideoFrame, self.block_contours, -1, (255, 0, 255), 3)

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        # 1. Take a copy of the raw depth frame
        depth_img = self.DepthFrameRaw.copy()
        
        # 2. Threshold depth to isolate objects resting ON the table
        # We assume the table sits at a higher depth value and blocks are closer.
        # Note: You may need to adjust these threshold values depending on actual physical setup
        mask = cv2.inRange(depth_img, 100, 950) 
        
        # 3. Clean up noise in the mask using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 4. Find the contours of the detected block blobs
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Save contours to the class attribute so they can be drawn
        self.block_contours = contours

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        modified_image = self.VideoFrame.copy()
        # Write your code here
        # Extract X and Y coordinates from the meshgrid
        X, Y = self.grid_points
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        
        # Assume Z is 0 (grid is flat on the board frame)
        Z_flat = np.zeros_like(X_flat)
        ones = np.ones_like(X_flat)
        
        # Create homogeneous 3D world points matrix (Shape: 4 x N)
        world_pts = np.vstack((X_flat, Y_flat, Z_flat, ones))
        
        # Transform points from World -> Camera frame using Extrinsic Matrix
        # Note: Using extrinsic_matrix_inv mapping standard convention World to Camera
        cam_pts = self.extrinsic_matrix @ world_pts 
        
        # Extract 3D points in the camera frame (Drop homogeneous 1 for perspective projection)
        cam_pts_3d = cam_pts[0:3, :]
        
        # Project into 2D pixel space using Intrinsic Matrix (Shape: 3 x N)
        pixels_homogenous = self.intrinsic_matrix @ cam_pts_3d
        
        # 1. Extract the Z values (depth)
        z_values = pixels_homogenous[2, :]
        
        # 2. Prevent divide-by-zero by ensuring Z is never exactly 0
        # (This replaces any 0 or negative Z with a tiny positive number)
        safe_z = np.maximum(z_values, 1e-5)
        
        # 3. Perform the true divide
        u = pixels_homogenous[0, :] / safe_z
        v = pixels_homogenous[1, :] / safe_z

        for i in range(len(u)):
            # 4. Skip drawing any points that are behind or inside the camera
            if z_values[i] <= 0.0:
                continue
                
            px, py = int(u[i]), int(v[i])
            
            # Make sure you use the updated bounds from earlier!
            if 0 <= px < 640 and 0 <= py < 480:
                cv2.circle(modified_image, (px, py), 4, (0, 255, 0), -1)

        self.GridFrame = modified_image


    
    def solve_extrinsic(self):
        """ Solve extrinsic matrix using detected board tags and their world coordinates
        Origin (0,0) is at top-left corner of the board"""
        
        obj_points_list = []
        img_points_list = []

        # detect all four board tags
        for detection in self.tag_detections:
            tag_id = detection.id
            if tag_id in self.boardTag_center:
                x,y = self.boardTag_center[tag_id]

                half = 25    # half tag size in mm
                # get the corner positions of the tag
                obj_points_list.extend([
                    [x-half, y-half, 0],
                    [x+half, y-half, 0],
                    [x-half, y+half, 0],
                    [x+half, y+half, 0]
                ])  # obj_points = measured world coordinates

                # img_points = detected pixel coordiantes of the tag 
                for corner in detection.corners:
                    img_points_list.append([corner.x, corner.y , 0])
        
        if (len(obj_points_list) <4):
            return None
        # convert obj_points and img_points to np.array float64 for openCV
        obj_points = np.array(obj_points_list, dtype=DTYPE)
        img_points = np.array(img_points_list, dtype=DTYPE)

        # use openCV solvePnP to get rotation and translation vectors
        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, self.intrinsic_matrix, self.dist_coeff, flags=cv2.SOLVEPNP_ITERATIVE)

        if success:
            # Convert rotation vector to 3x3 matrix
            R, _ = cv2.Rodrigues(rvec)
                
            # Create the 3x4 Extrinsic Matrix [R | t]
            extrinsic_matrix = np.hstack((R, tvec))
            self.extrinsic_matrix = extrinsic_matrix
            self.camera_calibrated = True   # both intrinsic and extrinsic calibration completed



    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """
        modified_image = self.VideoFrame.copy()
        # Write your code here
        # Check if msg is valid and contains detections
        if msg is not None and hasattr(msg, 'detections'):
            
            for detection in msg.detections:

                # 1. Draw Center Point
                cx = int(detection.centre.x)
                cy = int(detection.centre.y)
                cv2.circle(modified_image, (cx, cy), 5, (0, 0, 255), -1)
                
                # 2. Draw Tag ID
                tag_id = str(detection.id)
                cv2.putText(modified_image, f"ID: {tag_id}", (cx + 10, cy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # 3. Draw Tag Corners
                if hasattr(detection, 'corners'):
                    pts = np.array([[int(pt.x), int(pt.y)] for pt in detection.corners], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(modified_image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)


        self.TagImageFrame = modified_image

class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image


class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        if msg is not None and hasattr(msg, 'detection'):
            self.camera.tag_detections = msg

            if self.intrinsic_matrix is not None:   # intrinsic calibration data loaded
                self.solve_extrinsic()  

        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)



class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        # print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()