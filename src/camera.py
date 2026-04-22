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
import os
import sys
from pathlib import Path
import pyrealsense2 as rs
# import scipy.ndimage as ndimage
import math

current_dir = os.path.dirname(os.path.abspath(__file__))

yolo_src_path = os.path.abspath(os.path.join(current_dir, '..', 'yolo', 'src'))

if yolo_src_path not in sys.path:
    sys.path.append(yolo_src_path)

from model import *

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
        self.logger = rclpy.logging.get_logger('camera_helper')

        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = None # np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagDepthFrame = None
        self.comparison = None

        # mouse clicks & calibration variables
        self.camera_calibrated = False
        self.intrinsic_matrix = None
        self.extrinsic_matrix = None
        self.dist_coeff = np.array([0.130053, -0.216480, -0.002483, -0.006660, 0.000000])
        self.last_click = np.array([0, 0]) # This contains the last clicked position
        self.new_click = False # This is automatically set to True whenever a click is received. Set it to False yourself after processing a click
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = None
        self.tag_locations = [[-250, -25], [250, -25], [250, 275], [-250, 275]]
        """ block info """
        self.block_contours = []
        self.block_detections = {
            'uvds': [],
            'xyzs': [],
            'contours': [],
            'thetas': [],
            'colors': [],
            'sizes': [],
            'all_contours': [],
            'has_cluster': False
        }
        self.blocksize = 40     # for rounding measured height of block

        # April tag IDS and positions for building the board
        self.boardTag_center =  {  
            4: [-250, 275, 0],        # top-left
            3: [250, 275, 0],         # top-right
            1: [-250, -25, 0],        # bottom-left
            2: [250, -25, 0]          # bottom-right
        }   

        # homography transformation variables
        self.homography = False
        self.H = None
        self.H_inv = None

        # maximum depth from camera to board, used for calculating z
        self.max_depth = None

        # bin configurataion 
        self.bin_definitions = {
            "bin1": {
                "tag_ids": (5, 6),   
                "length": 220.0,       # mm
                "width": 150.0,        # mm
                "buffer_pad_length": 25.0,
                "buffer_pad_width": 10.0,
                "drop_length": 175.0,
                "drop_width": 100.0,
                "drop_offset": [0.0, 0.0, 0.0],
                "tag_to_bin_center": 110.0
            },
            "bin2": {
                "tag_ids": (7, 8),   
                "length": 220.0,
                "width": 150.0,
                "buffer_pad_length": 25.0,
                "buffer_pad_width": 10.0,
                "drop_length": 175.0,
                "drop_width": 100.0,
                "drop_offset": [0.0, 0.0, 0.0],
                "tag_to_bin_center": 110.0
            }
        }
        
        # stores latest computed bin / buffer / drop regions
        self.bin_rectangles = {}
        self.model = load_model()

    def run_autonomous(self, selected_blocks):
        return
        while True:
            blocks = find_target_blocks(self.model, self.VideoFrame, selected_blocks)
            if len(blocks) == 0:
                return
            print(blocks[0])

    def Homography_Transform(self, image):

        if self.extrinsic_matrix is None:
            return image

        width = 1000    # maintain the relative aspect ratio (1000x650)
        height = 650
        offset = 50     # add a bit of offset to display entire board
        dest_pts = {
            'topleft': [offset, offset],  
            'topright': [offset+width,offset],  
            'bottomleft': [offset,offset+height],   
            'bottomright': [offset+width,offset+height]    
        }
    
        # use world coordiantes of board's corners
        world_corners = {
           'topleft': [-500, 475],
           'topright': [500, 475],
           'bottomleft': [-500, -175],
           'bottomright': [500, -175]
        }

        ordered_keys = ['topleft', 'topright', 'bottomleft', 'bottomright']
        src_pts_list = []
        dest_pts_list = []

        for key in ordered_keys:
            src_world_x = world_corners[key][0]
            src_world_y = world_corners[key][1]
            src_pts_list.append(self.world_to_pixel(src_world_x, src_world_y))
            dest_pts_list.append((dest_pts[key][0], dest_pts[key][1]))


        if len(src_pts_list) !=4:
            return image

        src_arr = np.array(src_pts_list, dtype=DTYPE)
        dest_arr = np.array(dest_pts_list, dtype=DTYPE)

        H,_ = cv2.findHomography(src_arr, dest_arr)

        # write to file
        home_dir = str(Path.home())
        save_dir = os.path.join(home_dir, "robot_data")
        h_path = os.path.join(save_dir, "homography_matrix.txt")

        # base_path = os.path.dirname(os.path.abspath(__file__))
        # file_path = os.path.join(base_path, "extrinsic_matrix.txt")
        
        with open(h_path, "w") as f:
            f.write("Homography Matrix:\n")
            matrix_str = np.array2string(H, precision=4, suppress_small=True)
            f.write(matrix_str)
        self.H = H
        self.H_inv = np.linalg.inv(H)

       # new_img = cv2.warpPerspective(image, H, (1100, image.shape[0]))

        new_img = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))
        self.homography = True
        return new_img
    

    def undo_homography(self, u_warped, v_warped):
        """undo homography transformation to convert the pixel coordiantes to correct world coordinates"""

        # 1. Compute the Inverse Matrix
        
        # 2. Prepare the warped point in homogeneous coordinates
        point = np.array([u_warped, v_warped, 1.0]).reshape(3, 1)
        
        # 3. Transform back to raw pixel space
        raw_pt_h = self.H_inv @ point
        
        # 4. Perspective division (normalize the coordinates)
        u_raw = raw_pt_h[0] / raw_pt_h[2]
        v_raw = raw_pt_h[1] / raw_pt_h[2]
        
        return [float(u_raw), float(v_raw)]


    def world_to_warped_pixel(self, x_world, y_world):
        # Mapping World X [-500, 500] to Pixel [50, 1050]
        px = (x_world + 500) + 50
        
        # Mapping World Y [475, -175] to Pixel [50, 700]
        # (Top of board is 475, bottom is -175)
        py = (475 - y_world) + 50
        
        return px, py

    def round_to_blocksize(self, height):
        return round(height/self.blocksize)*self.blocksize


    def world_to_pixel(self, world_x, world_y, world_z=0):
        """
        convert world to pixel (u,v)
        """

        if self.extrinsic_matrix is None:
            return
        # 1. Create the 3D point in homogeneous coordinates [X, Y, Z, 1]
        world_point = np.array([[world_x], [world_y], [world_z], [1.0]])

        # 2. Apply Extrinsic Matrix (World -> Camera Space)
        # This accounts for the camera's exact rotation and position in the room
        camera_point = np.dot(self.extrinsic_matrix, world_point)

        # 3. Apply Intrinsic Matrix (Camera Space -> Image Plane)
        # We only use the [X, Y, Z] from the camera point (first 3 rows)
        pixel_homogeneous = np.dot(self.intrinsic_matrix, camera_point[:3])

        # 4. Perspective Division
        # This is critical: Z is the depth of the point relative to the camera lens.
        # Dividing by Z is what creates the perspective effect (further away = smaller).
        z_c = pixel_homogeneous[2, 0]
        
        if abs(z_c) < 1e-6:
            return None # Point is at or behind the camera lens
        
        u = pixel_homogeneous[0, 0] / z_c
        v = pixel_homogeneous[1, 0] / z_c

        return (int(round(u)), int(round(v)))
    

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
        if (self.DepthFrameRaw is None):
            return 
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

        

       
    def blockDetector(self):
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(self.VideoFrame, cv2.COLOR_RGB2GRAY)
        
        # 2. Use a more robust Threshold (Otsu's method is better than a fixed 127)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Clear old detections
        self.block_detections.reset()

        for contour in contours:
            # Calculate Area to filter out noise
            area = cv2.contourArea(contour)
            if area < 500: # Adjust based on your camera height
                continue

            # 4. Calculate Center (Moments)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 5. Get Depth (Z) at this center point
                # Ensure cx, cy are within bounds
                cz = self.DepthFrameRaw[cy, cx]

                # 6. Convert Pixel (u, v, d) to World (X, Y, Z)
                # You likely have a function for this already
                world_coords = self.pixel_to_World(cx, cy, cz)

                # 7. Store the detection
                self.block_detections['uvds'].append([cx, cy, cz])
                self.block_detections['xyzs'].append(world_coords)
                self.block_detections['contours'].append(contour)
                
                # Optional: Calculate rotation
                rect = cv2.minAreaRect(contour)
                theta = rect[2]
                self.block_detections['thetas'].append(np.deg2rad(theta))

        # 8. Visual Feedback
        self.block_contours = contours
        cv2.drawContours(self.VideoFrame, self.block_contours, -1, (255, 0, 255), 3)
     #   cv2.imshow('')
        



    def detectBlocksInDepthImage(self, _lower=700, _upper=960, blind_rect=None, sort_key="color"):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        self.block_detections.reset()
        lower = _lower
        upper = _upper
        """mask out arm & outside board"""
        # self.ProcessDepthFrameRaw = cv2.GaussianBlur(self.ProcessDepthFrameRaw, (5, 5), 3)
        self.ProcessDepthFrameRaw = cv2.medianBlur(self.ProcessDepthFrameRaw, 3)
        mask = np.zeros_like(self.ProcessDepthFrameRaw, dtype=np.uint8)
        # !!! Attention to these rectangles's range
        cv2.rectangle(mask, (225, 90), (1083, 700), 255, cv2.FILLED)
        cv2.rectangle(mask, (570, 400),(735, 700), 0, cv2.FILLED)
        if blind_rect is not None:
            cv2.rectangle(mask, blind_rect[0], blind_rect[1], 0, cv2.FILLED)

        depth_seg = cv2.inRange(self.ProcessDepthFrameRaw, lower, upper)
        img_depth_thr = cv2.bitwise_and(depth_seg, mask)

        contours, _ = cv2.findContours(img_depth_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        self.block_detections['all_contours'] = contours


        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] < 200 or abs(M["m00"]) > 7000:
                # reject false positive detections by area size
                continue
            mask_single = np.zeros_like(self.ProcessDepthFrameRaw, dtype=np.uint8)
            cv2.drawContours(mask_single, [contour], -1, 255, cv2.FILLED)
            depth_single = cv2.bitwise_and(self.ProcessDepthFrameRaw, self.ProcessDepthFrameRaw, mask=mask_single)
            depth_array = depth_single[depth_single>=lower]

            # Stats mode range
            mode_real, _ = stats.mode(depth_array)
            # print("real mode", mode_real)
            depth_diff =  mode_real - depth_array
            depth_array_inliers = depth_array[depth_diff<8]

            # Inter Quartile Range
            # Q1 = np.percentile(depth_array, 25, interpolation = 'midpoint')
            # Q3 = np.percentile(depth_array, 75, interpolation = 'midpoint')
            # IQR = Q3 - Q1
            # mode_lower = Q1 - 1.5 * IQR # outlier lower bound
            # print("IQR lower", mode_lower)
            # depth_array_inliers = depth_array[depth_array>=mode_lower]
            
            mode = np.min(depth_array_inliers)
            # mode = np.min(depth_array)
            # print("result min", mode)
            # !!! Attention to the mode offset, it determines how much of the top surface area will be reserved
            depth_new = cv2.inRange(depth_single, lower, int(mode)+5)
            contours_new, _ = cv2.findContours(depth_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            if not contours_new:
                continue
            contours_new_valid = max(contours_new, key=cv2.contourArea) # find the largest contour
            M = cv2.moments(contours_new_valid)

            if abs(M["m00"]) < 200:
                # reject false positive detections by area size
                continue
            elif abs(M["m00"]) > 2000:
                # TODO add seg model
                print("Cluster detected with moment:", M["m00"])
                self.block_detections.reset()
                self.block_detections['all_contours'] = contours
                self.block_detections['has_cluster'] = True
                # # generate new mask for new valid contours
                # mask_new_single = np.zeros_like(mask_single, dtype=np.uint8)
                # cv2.drawContours(mask_new_single, [contours_new_valid], -1, 255, cv2.FILLED)
                # # segmente rgb image using new mask
                # rgb_single = cv2.bitwise_and(self.ProcessVideoFrame, self.ProcessVideoFrame, mask=mask_new_single)
                # input_img = BlocksDataset.transform(torch.from_numpy(rgb_single).to(torch.float).permute(2, 0, 1)).unsqueeze(0)
                # # input_img (1, 3, 244, 244)
                # output_pred = self.model(input_img.to(self.device))
                # # output_pred (1, 7, 244, 244)
                # output = torch.argmax(output_pred, 1).squeeze(0).cpu().numpy()
                # # output (244, 244) int64
                # bins = np.bincount(output.flatten())
                # if np.count_nonzero(bins[1:])>1:
                #     output_img = output.astype(np.float32) * 255/6
                #     output_mask = cv2.resize(output_img , (1280,720))
                #     print("Your model really find something??!!")
                #     print("model colors:{}".format(bins[1:]))
                #     cv2.imwrite("data/treasures_%d.png" % (random()*1000), output_mask)
                # pass
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cz = self.ProcessDepthFrameRaw[cy, cx]
            block_ori = - cv2.minAreaRect(contours_new_valid)[2] # turn the range from [-90, 0) to (0, 90]
            # print(block_ori)

            block_xyz = self.pixel_to_World(cx, cy, cz)
              # !!! size classification: attention to this moment threshold
            if M["m00"] < 850:
                block_xyz[2] = block_xyz[2] - 12.5
                self.block_detections['sizes'].append(1) # 1 for small
            else:
                block_xyz[2] = block_xyz[2] - 19
                self.block_detections['sizes'].append(0) # 0 for large

            self.block_detections['uvds'].append([cx, cy, cz])
            self.block_detections['xyzs'].append(block_xyz)
            self.block_detections['contours'].append(contours_new_valid)
            self.block_detections['thetas'].append(np.deg2rad(block_ori))
            self.block_detections['colors'].append(self.retrieve_area_color(self.ProcessVideoFrame, self.ProcessVideoFrameLab, self.ProcessVideoFrameHSV, contours_new_valid))
            
            # print(self.color_id[self.block_detections.colors[-1]], M["m00"])
            if self.block_detections['has_cluster']:
                break

        self.block_detections.update(sort_key)
        alignment_frame = self.VideoFrame.copy()

       # 2. Draw RGB contours in NEON BLUE
        # These are from your blockDetector() function
        if hasattr(self, 'block_contours') and self.block_contours is not None:
            cv2.drawContours(alignment_frame, self.block_contours, -1, (255, 255, 0), 2)

        # 3. Draw DEPTH contours in NEON MAGENTA
        # These are the ones you just found in the depth image
        if self.block_detections['contours']:
            cv2.drawContours(alignment_frame, self.block_detections['contours'], -1, (255, 0, 255), 2)

        # 4. Add a legend so you know which is which
        cv2.putText(alignment_frame, "CYAN: RGB | MAGENTA: Depth", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 5. Display the comparison
        # cv2.imshow("Sensor Alignment Check", alignment_frame)
        # cv2.waitKey(1)


    def drawAlignmentComparison(self):
        # 1. Use the current RGB frame as the background
        canvas = self.VideoFrame.copy()

        # 2. Draw RGB contours (The "Visual" guess) in NEON CYAN
        if hasattr(self, 'block_contours') and self.block_contours is not None:
            cv2.drawContours(canvas, self.block_contours, -1, (255, 255, 0), 2)

        # 3. Draw Depth contours (The "Physical" truth) in NEON MAGENTA
        if hasattr(self, 'block_detections') and self.block_detections['contours']:
            cv2.drawContours(canvas, self.block_detections['contours'], -1, (255, 0, 255), 2)

        # 4. Add a legend for clarity
        cv2.putText(canvas, "CYAN: RGB | MAGENTA: Depth", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 5. Show the window
        return canvas




    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        if (self.extrinsic_matrix is None):
            return
        
        
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

        if self.homography:
            # Prepare points for cv2: shape (N, 1, 2)
            pts_to_warp = np.array([u, v]).T.reshape(-1, 1, 2).astype(np.float32)
            # This moves the dots into the Top-Down view
            warped_pts = cv2.perspectiveTransform(pts_to_warp, self.H)
            # Now we loop through the warped points specifically
            for i in range(len(warped_pts)):
                # Extract px, py from the warped result
                px, py = warped_pts[i][0]
                
                # Draw on the warped image
                cv2.circle(modified_image, (int(px), int(py)), 4, (0, 255, 0), -1)
        else:
            # Fallback: Draw on the normal tilted frame
            for i in range(len(u)):
                px, py = int(u[i]), int(v[i])
                cv2.circle(modified_image, (px, py), 4, (0, 255, 0), -1)
            
            # # Make sure you use the updated bounds from earlier!
            # if -450 <= px <= 500 and -175 <= py <= 525:
            cv2.circle(modified_image, (px, py), 4, (0, 255, 0), -1)

        self.GridFrame = modified_image



    def get_maxDepth(self):
        if (self.extrinsic_matrix is None):
            return

        
        # get the pixel coordinates of April Tag
        tag_world = np.array([
            [-250,  275, 0, 1],  # Tag 4
            [ 250,  275, 0, 1],  # Tag 3
            [-250, -25,  0, 1],  # Tag 1
            [ 250, -25,  0, 1]   # Tag 2
        ]).T # Transpose to (4, 4) for matrix math

        # 2. Project World -> Camera Frame -> Pixel Space
        # This uses the current extrinsic_matrix, so it updates as the camera moves!
        cam_corners = self.extrinsic_matrix @ tag_world
        pixel_corners_h = self.intrinsic_matrix @ cam_corners[0:3, :]
        
        # 3. Divide by Z to get final (u, v) pixels
        # Using a small epsilon to prevent division by zero
        z_coords = np.maximum(pixel_corners_h[2, :], 1e-5)
        u_corners = pixel_corners_h[0, :] / z_coords
        v_corners = pixel_corners_h[1, :] / z_coords

        # 4. Save these as your src_pts for Homography
        # We stack them into a (4, 2) array of float32
        tag_pixels = np.vstack((u_corners, v_corners)).T.astype(np.float32)
        max_depth =0
        for pt in tag_pixels:
            u,v = pt
            u_int = int(u)
            v_int = int(v)
            if (self.DepthFrameRaw[v_int][u_int] > max_depth):
                max_depth = self.DepthFrameRaw[v_int][u_int]
        
        self.max_depth = max_depth
        print(f"Max depth: {max_depth}")




    def coord_pixel_to_world(self, u, v, d):
        '''
        Convert pixel coordinates (from camera frame) to world coordinates
        u: pixel x coordinate
        v: pixel y coordinate
        depth: depth value at pixel (u, v) # TODO: find its units
        '''
        if self.extrinsic_matrix is None:
            print("extrinsic matrix is undefined")
            return
        
        if (self.max_depth is None):
            return
        
        if d > self.max_depth:
            z = 0   # z must be positive
        else:
            z = self.max_depth - d
 
        print("original z from depth frame: ", z)

        index = np.array([u, v, 1]).reshape((3,1))
        pos_camera = d * np.matmul(self.intrinsic_matrix_inv, index)
        temp_pos = np.array([pos_camera[0][0], pos_camera[1][0], pos_camera[2][0], 1]).reshape((4,1))
        extrinsic_matrix_inv = np.linalg.inv(self.extrinsic_matrix)
        world_pos = np.matmul(extrinsic_matrix_inv, temp_pos)
    
        pos = world_pos.flatten()[:3]
        print(f"original height z: {pos[2]}")

        offset = 15
        roi = self.DepthFrameRaw[v-offset:v+offset, u-offset:u+offset]
        # smoothed_roi = ndimage.median_filter(roi, size=3)
        valid_depths = roi[(roi > 0)]
        min_depth = np.min(valid_depths)
        min_depth = np.percentile(valid_depths, 5)
        d= min_depth

        z = self.max_depth - min_depth
        print(f"min z around the pixel: {z}")
        
        index = np.array([u, v, 1]).reshape((3,1))
        pos_camera = d * np.matmul(self.intrinsic_matrix_inv, index)
        temp_pos = np.array([pos_camera[0][0], pos_camera[1][0], pos_camera[2][0], 1]).reshape((4,1))
        extrinsic_matrix_inv = np.linalg.inv(self.extrinsic_matrix)
        world_pos = np.matmul(extrinsic_matrix_inv, temp_pos)

        pos = world_pos.flatten()[:3]
        pos[2] = self.round_to_blocksize(pos[2])
       # print(f"worldX: {pos[0]}, worldY: {pos[1]}, worldZ: {pos[2]}")     

        return pos

    


    def pixel_to_World(self, u, v,d):
        """Convert Pixel coordinates to World using extrinsic matrix """
        if self.extrinsic_matrix is None: 
            print("extrinsic matrix is undefined")
            return None

        pixel_vector = np.array([[u],[v], [1]])

        camera_ray = self.intrinsic_matrix_inv @ pixel_vector

        # 2. Invert the Extrinsic Matrix to get the Camera-to-World transformation
        R = self.extrinsic_matrix[:3, :3]
        t = self.extrinsic_matrix[:3, 3:]
        
        R_inv = R.T              # Inverse of a rotation matrix is its transpose
        t_inv = -R_inv @ t       # Physical location of the camera lens in the World
        
        # 3. Rotate the camera ray into the World orientation
        world_ray = R_inv @ camera_ray
        
        # 4. Find where the ray hits the board (Line-Plane Intersection)
        scale_factor = -t_inv[2, 0] / world_ray[2, 0]
        
        # 5. Plug the scale factor back in to get the exact X and Y world coordinates
        world_point = t_inv + (scale_factor * world_ray)

        
        # use the camera pinhole model to get height z
        pos = self.coord_pixel_to_world(u, v, d)
        if pos is None:
            return
        world_point[2,0] = pos[2]
    
        print(f"worldX: {world_point[0, 0]}, worldY: {world_point[1, 0]}, worldZ: {world_point[2,0]}") 

        return [world_point[0, 0], world_point[1, 0], world_point[2,0]] # invert y axis to align with motor direction



    def solve_extrinsic(self):
        """ Solve extrinsic matrix using detected board tags and their world coordinates
        Origin (0,0) is at the robot's position. Use boardTag_center for the position of April tags"""

        # if self.tag_detections is None or not hasattr(self.tag_detections, 'detections'):
        #     return
        if self.intrinsic_matrix is None:
            return


        obj_points_list = []
        img_points_list = []

        # detect all four board tags
        for detection in self.tag_detections:
            tag_id = detection.id
            if tag_id in self.boardTag_center:
                x,y,z = self.boardTag_center[tag_id]

                # obj_points = measured world coordinates
                obj_points_list.append([x,y,z])

                # img_points = detected pixel coordiantes of the tag 
                img_points_list.append([detection.centre.x,detection.centre.y])
                # debug_img = self.VideoFrame.copy()
                # window_name = f"Debug_Tag_{tag_id}"
                # cv2.imshow(window_name, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(1)
                    
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
                    
            # Create the 4x4 Extrinsic Matrix [R | t]
            extrinsic_matrix = np.eye(4, dtype=DTYPE)
            extrinsic_matrix[:3, :3] = R
            extrinsic_matrix[:3, 3] = tvec.squeeze()
            
            self.extrinsic_matrix = extrinsic_matrix
            self.camera_calibrated = True   # both intrinsic and extrinsic calibration completed
            print("extrinsic matrix solved successfully:")
            try:
                # debug_img = self.VideoFrame.copy()
                # window_name = f"Debug_Tag_{tag_id}"
                # cv2.imshow(window_name, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(1)
                home_dir = str(Path.home())
                save_dir = os.path.join(home_dir, "robot_data")
                extrinsic_path = os.path.join(save_dir, "extrinsic_matrix.txt")
                intrinsic_path = os.path.join(save_dir, "intrinsic_matrix.txt")

                # base_path = os.path.dirname(os.path.abspath(__file__))
                # file_path = os.path.join(base_path, "extrinsic_matrix.txt")
                
                with open(extrinsic_path, "w") as f:
                    f.write("Extrinsic Matrix (World to Camera):\n")
                    matrix_str = np.array2string(extrinsic_matrix, precision=4, suppress_small=True)
                    f.write(matrix_str)
   
                
                print("Successfully saved extrinsic matrix to extrinsic_matrix.txt")
            except Exception as e:
                print(f"Failed to write to file: {str(e)}")

    
    def compareContours(self, msg):
        # 1. Create two black backgrounds (Grayscale)
        # Using the shape of your current video frame
        if self.extrinsic_matrix is None:
            return
        
        h, w = self.VideoFrame.shape[:2]
        mask_rgb = np.zeros((h, w), dtype=np.uint8)
        mask_depth = np.zeros((h, w), dtype=np.uint8)

        if msg is not None and hasattr(msg, 'detections'):
            for detection in msg.detections:
                # Prepare corner points for OpenCV
                pts = np.array([[int(pt.x), int(pt.y)] for pt in detection.corners], np.int32)
                
                # Draw the RGB contour on mask_rgb
                # We draw a thick outline (thickness=2)
                cv2.polylines(mask_rgb, [pts], True, 255, 1, cv2.LINE_AA)
                
                # Draw the Depth contour on mask_depth
                # Note: Since the detection comes from the RGB frame, 
                # we are checking if the Depth data at these same pixels aligns.
                # We use the same corners to see if they land on the depth-shadow/edges.
                cv2.polylines(mask_depth, [pts], True, 255, 1, cv2.LINE_AA)

        # 2. Create a Color Comparison Frame (BGR)
        # We will put RGB detection in the Blue channel and Depth in the Green channel
        comparison = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Blue Channel = RGB Detection
      #  comparison[:, :, 0] = mask_rgb 
        # Green Channel = Depth Detection (logic check)
        comparison[:, :, 1] = mask_depth 

        # 3. Add a "True" Depth Edge to compare
        # This is the most important part: drawing the ACTUAL edges from the depth sensor
        depth_8bit = cv2.convertScaleAbs(self.DepthFrameRGB, alpha=0.03)
        depth_edges = cv2.Canny(depth_8bit, 100, 200)
        
        # Put actual depth sensor edges in RED
        comparison[:, :, 2] = depth_edges
        print("in Compare Contours")

        # 4. Resulting Logic:
        # If the Red lines (actual depth) sit perfectly on the Green lines (RGB detection), 
        # your alignment is 100% correct.
        self.comparison = comparison




    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """

        if self.DepthFrameRaw is None:
            return
        
        modified_image = self.VideoFrame.copy()
        # We use cv2.applyColorMap to make the depth data visible to humans
        depth_color = cv2.applyColorMap(
        cv2.convertScaleAbs(self.DepthFrameRaw, alpha=0.03), 
        cv2.COLORMAP_JET)

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

                cv2.circle(depth_color, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(depth_color, f"ID: {tag_id}", (cx + 10, cy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.polylines(depth_color, [pts], True, (0, 255, 0), 2)
            #    sys.exit("april tag callback")


        self.TagImageFrame = modified_image
        self.TagDepthFrame = depth_color  # Store the processed depth frame

    def get_depth_at_pixel(self, u, v, window=7):
        """
        Robust depth estimate near a pixel using median of valid depths.
        """
        if self.DepthFrameRaw is None:
            return None

        h, w = self.DepthFrameRaw.shape[:2]
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))

        half = window // 2
        u0 = max(0, u - half)
        u1 = min(w, u + half + 1)
        v0 = max(0, v - half)
        v1 = min(h, v + half + 1)

        roi = self.DepthFrameRaw[v0:v1, u0:u1]
        valid = roi[roi > 0]

        if valid.size == 0:
            return None

        return float(np.median(valid))


    def get_tag_center_world(self, detection):
        """
        Convert the AprilTag center from image pixel coordinates to world coordinates.
        """
        cx = int(detection.centre.x)
        cy = int(detection.centre.y)

        d = self.get_depth_at_pixel(cx, cy, window=7)
        if d is None:
            return None

        return self.pixel_to_World(cx, cy, d)


    def get_tag_bottom_edge_world(self, detection):
        """
        Get the bottom edge of a visible tag in world coordinates.

        Assumes AprilTag corners are ordered consistently as:
            0 = top-left
            1 = top-right
            2 = bottom-right
            3 = bottom-left

        Then the bottom edge is corner 3 -> corner 2.

        """
        if not hasattr(detection, "corners") or len(detection.corners) < 4:
            return None

        c_bl = detection.corners[3]   # bottom-left
        c_br = detection.corners[2]   # bottom-right

        d_bl = self.get_depth_at_pixel(int(c_bl.x), int(c_bl.y), window=5)
        d_br = self.get_depth_at_pixel(int(c_br.x), int(c_br.y), window=5)

        if d_bl is None or d_br is None:
            return None

        p_bl = self.pixel_to_World(int(c_bl.x), int(c_bl.y), d_bl)
        p_br = self.pixel_to_World(int(c_br.x), int(c_br.y), d_br)

        if p_bl is None or p_br is None:
            return None

        return np.array(p_bl, dtype=float), np.array(p_br, dtype=float)


    def build_oriented_rectangle(self, center_xy, theta, length, width, z=0.0):
        """
        Build the 4 world-coordinate corners of an oriented rectangle.
        theta defines the local +x axis.
        """
        u = np.array([math.cos(theta), math.sin(theta)])      # local x-axis
        v = np.array([-math.sin(theta), math.cos(theta)])     # local y-axis

        half_L = 0.5 * length
        half_W = 0.5 * width

        c1_xy = center_xy - half_L * u - half_W * v
        c2_xy = center_xy + half_L * u - half_W * v
        c3_xy = center_xy + half_L * u + half_W * v
        c4_xy = center_xy - half_L * u + half_W * v

        return [
            [c1_xy[0], c1_xy[1], z],
            [c2_xy[0], c2_xy[1], z],
            [c3_xy[0], c3_xy[1], z],
            [c4_xy[0], c4_xy[1], z],
        ]


    def estimate_bin_pose_from_single_tag(self, detection, bin_config):
        """
        Estimate bin pose from one visible tag using the fact that
        the bin always lies along the bottom border of the tag.

        Returns:
            {
                "center_xy": ...,
                "center_z": ...,
                "theta": ...,
                "source": "single_tag_bottom_edge"
            }
        """
        tag_world = self.get_tag_center_world(detection)
        if tag_world is None:
            return None

        bottom_edge = self.get_tag_bottom_edge_world(detection)
        if bottom_edge is None:
            return None

        p_bl, p_br = bottom_edge

        # bin width direction follows the tag bottom edge
        edge_vec = p_br[:2] - p_bl[:2]
        edge_norm = np.linalg.norm(edge_vec)
        if edge_norm < 1e-6:
            return None

        width_dir = edge_vec / edge_norm

        # normal candidates to the bottom edge
        normal1 = np.array([-width_dir[1], width_dir[0]])
        normal2 = -normal1

        tag_center_xy = np.array(tag_world[:2], dtype=float)
        edge_mid_xy = 0.5 * (p_bl[:2] + p_br[:2])

        # choose the normal that points from the bottom edge toward the tag center
        to_center = tag_center_xy - edge_mid_xy
        if np.dot(normal1, to_center) > np.dot(normal2, to_center):
            inward_normal = normal1
        else:
            inward_normal = normal2

        offset = float(bin_config["tag_to_bin_center"])
        center_xy = tag_center_xy + offset * inward_normal
        center_z = float(tag_world[2])

        # local +x axis points inward from tag toward bin center
        theta = math.atan2(inward_normal[1], inward_normal[0])

        return {
            "center_xy": center_xy,
            "center_z": center_z,
            "theta": theta,
            "source": "single_tag_bottom_edge"
        }


    def find_bin_rectangles_from_tags(self):
        """
        Compute bin, buffer, and drop zones.

        Priority:
          1) two visible tags
          2) one visible tag using the bottom border of the tag
        """
        self.bin_rectangles = {}

        if self.tag_detections is None:
            print("tag detections not initialized")
            return self.bin_rectangles

        if self.extrinsic_matrix is None or self.intrinsic_matrix is None:
            print("extrinsic matrix not computed")
            return self.bin_rectangles

        tag_lookup = {det.id: det for det in self.tag_detections}

        for bin_name, config in self.bin_definitions.items():
            tag1_id, tag2_id = config["tag_ids"]
            pose = None

            # Case 1: both tags visible
            if tag1_id in tag_lookup and tag2_id in tag_lookup:
                p1 = self.get_tag_center_world(tag_lookup[tag1_id])
                p2 = self.get_tag_center_world(tag_lookup[tag2_id])

                if p1 is not None and p2 is not None:
                    p1 = np.array(p1, dtype=float)
                    p2 = np.array(p2, dtype=float)

                    center_xy = 0.5 * (p1[:2] + p2[:2])
                    center_z = 0.5 * (p1[2] + p2[2])

                    axis_vec = p2[:2] - p1[:2]
                    axis_norm = np.linalg.norm(axis_vec)

                    if axis_norm >= 1e-6:
                        theta = math.atan2(axis_vec[1], axis_vec[0])
                        pose = {
                            "center_xy": center_xy,
                            "center_z": center_z,
                            "theta": theta,
                            "source": "two_tags"
                        }

            # Case 2: only one tag visible
            if pose is None:
                if tag1_id in tag_lookup:
                    pose = self.estimate_bin_pose_from_single_tag(
                        tag_lookup[tag1_id],
                        config
                    )
                elif tag2_id in tag_lookup:
                    pose = self.estimate_bin_pose_from_single_tag(
                        tag_lookup[tag2_id],
                        config
                    )

            if pose is None:
                continue

            center_xy = pose["center_xy"]
            center_z = pose["center_z"]
            theta = pose["theta"]

            bin_length = float(config["length"])
            bin_width = float(config["width"])

            buffer_length = bin_length + 2.0 * float(config["buffer_pad_length"])
            buffer_width = bin_width + 2.0 * float(config["buffer_pad_width"])

            drop_length = float(config["drop_length"])
            drop_width = float(config["drop_width"])

            u = np.array([math.cos(theta), math.sin(theta)])
            v = np.array([-math.sin(theta), math.cos(theta)])

            dx, dy, dz = config["drop_offset"]
            drop_center_xy = center_xy + dx * u + dy * v
            drop_center_z = center_z + dz

            bin_corners = self.build_oriented_rectangle(
                center_xy, theta, bin_length, bin_width, center_z
            )

            buffer_corners = self.build_oriented_rectangle(
                center_xy, theta, buffer_length, buffer_width, center_z
            )

            drop_corners = self.build_oriented_rectangle(
                drop_center_xy, theta, drop_length, drop_width, drop_center_z
            )
           # print("here")

            self.bin_rectangles[bin_name] = {
                "tag_ids": (tag1_id, tag2_id),
                "center": [center_xy[0], center_xy[1], center_z],
                "theta": theta,
                "source": pose["source"],
                "bin": {
                    "length": bin_length,
                    "width": bin_width,
                    "corners": bin_corners
                },
                "buffer": {
                    "length": buffer_length,
                    "width": buffer_width,
                    "corners": buffer_corners
                },
                "drop": {
                    "center": [drop_center_xy[0], drop_center_xy[1], drop_center_z],
                    "length": drop_length,
                    "width": drop_width,
                    "corners": drop_corners
                }
            }

        return self.bin_rectangles


    def draw_zone_rectangle(self, image, corners_world, color, label=None):
        """
        Draw one world-frame rectangle on an image.
        """
        if image is None:
            return image

        pixel_pts = []

        for corner in corners_world:
            px = self.world_to_pixel(corner[0], corner[1], corner[2])
            if px is not None:
                pixel_pts.append(px)

        if len(pixel_pts) == 4:
            pts = np.array(pixel_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, color, 2)

            if label is not None:
                cx = int(np.mean([p[0] for p in pixel_pts]))
                cy = int(np.mean([p[1] for p in pixel_pts]))
                cv2.putText(image, label, (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image


    def draw_bin_regions_on_image(self, image):
        """
        Draw bin, buffer, and drop regions on image.
        """
        if image is None:
            return image

        output = image.copy()

        for bin_name, info in self.bin_rectangles.items():
            output = self.draw_zone_rectangle(
                output,
                info["bin"]["corners"],
                (0, 255, 0),
                f"{bin_name}_bin"
            )

            output = self.draw_zone_rectangle(
                output,
                info["buffer"]["corners"],
                (0, 255, 255),
                f"{bin_name}_buffer"
            )

            output = self.draw_zone_rectangle(
                output,
                info["drop"]["corners"],
                (255, 0, 0),
                f"{bin_name}_drop"
            )

            center = info["center"]
            center_px = self.world_to_pixel(center[0], center[1], center[2])
            if center_px is not None:
                cv2.circle(output, center_px, 5, (255, 255, 255), -1)
                cv2.putText(output, info["source"], (center_px[0] + 8, center_px[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        return output


    def is_point_in_oriented_rectangle(self, point_xyz, center_xyz, theta, length, width):
        """
        Check whether a world point lies inside an oriented rectangle.
        """
        px, py = point_xyz[0], point_xyz[1]
        cx, cy = center_xyz[0], center_xyz[1]

        rel = np.array([px - cx, py - cy], dtype=float)

        u = np.array([math.cos(theta), math.sin(theta)])
        v = np.array([-math.sin(theta), math.cos(theta)])

        proj_u = np.dot(rel, u)
        proj_v = np.dot(rel, v)

        return (abs(proj_u) <= length / 2.0) and (abs(proj_v) <= width / 2.0)


    def point_in_bin_region(self, point_xyz, bin_name, region="bin"):
        """
        region can be 'bin', 'buffer', or 'drop'
        """
        if bin_name not in self.bin_rectangles:
            return False

        info = self.bin_rectangles[bin_name]
        theta = info["theta"]

        if region == "drop":
            center = info["drop"]["center"]
            length = info["drop"]["length"]
            width = info["drop"]["width"]
        else:
            center = info["center"]
            length = info[region]["length"]
            width = info[region]["width"]

        return self.is_point_in_oriented_rectangle(point_xyz, center, theta, length, width)


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

       # self.camera.VideoFrame = self.camera.Homography_Transform(cv_image)
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
    
        # if self.camera.intrinsic_matrix is not None:   # intrinsic calibration data loaded
        #     self.camera.solve_extrinsic()  
    
        if np.any(self.camera.VideoFrame != 0):
            if hasattr(msg, 'detections'):
                self.camera.tag_detections = msg.detections

                if self.camera.extrinsic_matrix is None: 
                    self.camera.solve_extrinsic()

               # self.camera.find_bin_rectangles_from_tags()

            self.camera.drawTagsInRGBImage(msg)
            #self.camera.compareContours(msg)
        #    self.camera.detectBlocksInDepthImage(msg)

         #   self.camera.TagImageFrame = self.camera.draw_bin_regions_on_image(
            #    self.camera.TagImageFrame
           # )






class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        self.camera.intrinsic_matrix_inv = np.linalg.pinv(self.camera.intrinsic_matrix)



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
        # CASTROPHIC TO APPLY HOMOGRAPHY ON DEPTH FRAME
        #self.camera.DepthFrameRaw = self.camera.Homography_Transform(cv_depth)
        self.camera.DepthFrameRaw = cv_depth
        if (self.camera.max_depth is None):
            self.camera.get_maxDepth()
            
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

        # pipeline = rs.pipeline()
        # config = rs.config()
        # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # pipeline.start(config)

        # align_to = rs.stream.color
        # align = rs.align(align_to)

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
                        "RGBTag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow(
                        "DepthTag window",
                        cv2.cvtColor(self.camera.TagDepthFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Tag Alignment (Blue:RGB, Red:Actual Depth)", self.camera.comparison)
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
    # res = run_model()
    # metrics = eval_model()
    # print(res)
    # print(metrics.box.map)
    
    main()