import cv2
import numpy as np
import math

# HSV ranges for each color
hsv_ranges = {
    'black': {
        'lower': (0,0,0),
        'upper':(180,60,50)
    },
    'gray': {
        'lower': (0,0,50),
        'upper': (180, 40, 200)
    },
    'red':{
        'lower1': (0, 100, 100), 'upper1': (5, 255, 255),
        'lower2': (160, 80, 80), 'upper2': (180, 255, 255)   
    },
    'orange': {
        'lower': (6, 100, 100),
        'upper': (15,255,255)
    },
    'yellow': {
        'lower': (20,80,80),
        'upper': (40, 255,255)
    },
    'blue': {
        'lower': (95,100,50),
        'upper': (120, 255,255)
    },
    'green': {
        'lower': (30, 50,50),
        'upper': (95, 255, 255),
    },
    'purple': {
        'lower': (120, 40, 20),
        'upper': (155, 255, 255)
    },
    'brown': {
        'lower': (5, 60, 20),
        'upper': (25,200,120)
    }
}

def find_block(image, u, v):
    # search in a bounding region for this block's color
    # convert from rgb to hsv
    h_img, w_img = image.shape[:2]
    y1, y2 = max(0, v-10), min(h_img, v+10)
    x1, x2 = max(0, u-10), min(w_img, u+10)
    
    roi = image[y1:y2, x1:x2]


    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    max_pixels = 0
    color = None

    # find the color that has the most pixels in this region
    for color_name, bounds in hsv_ranges.items():
        if color_name == 'red':
            m1 = cv2.inRange(hsv, bounds['lower1'], bounds['upper1'])
            m2 = cv2.inRange(hsv, bounds['lower2'], bounds['upper2'])
            mask = cv2.bitwise_or(m1, m2)
        else:
            mask = cv2.inRange(hsv, bounds['lower'], bounds['upper'])


        pixel_count = cv2.countNonZero(mask)

        print(f"pixel count {pixel_count} for color {color_name}")

        if pixel_count > max_pixels:
            max_pixels = pixel_count
            color = color_name

    return color



def detect_uniqueColors(frame, color, u= None, v=None):
    # apply color masks
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    if color == 'red':
        mask1 = cv2.inRange(hsv, 
                            np.array(hsv_ranges['red']['lower1']), 
                            np.array(hsv_ranges['red']['upper1']))
        mask2 = cv2.inRange(hsv,
                            np.array(hsv_ranges['red']['lower2']), 
                            np.array(hsv_ranges['red']['upper2']))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, hsv_ranges[color]['lower'], hsv_ranges[color]['upper'])


    # Opening removes small noise; Closing fills small holes in the block
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    
    # 1. Pre-processing
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # # 2. Edge Detection (Canny)
    # # You may need to tune 50 and 150 based on your lighting
    # edges = cv2.Canny(blurred, 50, 150)
    
    # # 3. Morphological closing to join gaps in edges
    # kernel = np.ones((5,5), np.uint8)
    # closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 4. Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # check that the pixel is inside the contour
        is_inside = True
        if u is not None:
            is_inside = cv2.pointPolygonTest(cnt, (float(u), float(v)), False)
            
        if (is_inside):
            # filter out small noise by area
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
                
            # 5. Get the Minimum Area Rectangle (the "Oriented Bounding Box")
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=int)

            # order the corners from increasing y
            sorted_indices = np.argsort(box[:, 1])
            top_points = box[sorted_indices[2:]] # The two points with largest Y
            p1, p2 = top_points[np.argsort(top_points[:, 0])]

            # 3. Calculate Angle
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]

            # This gives angle in radians, convert to degrees
            # Since we want 0 to be vertical (Y-axis), we use atan2(dx, dy)
            angle_rad = math.atan2(dx, dy)
            angle_deg = math.degrees(angle_rad)
            if angle_deg <= 90:
                angle_deg = 90 - angle_deg
            else:
                angle_deg = angle_deg - 90
                angle_deg = -angle_deg


            # 6. Extract Orientation Data
            (x, y), (w, h), angle = rect
            
            # Adjust angle to be intuitive (OpenCV angles can be tricky)
            #if w < h:
            angle = angle
            # else:

           # angle = 90 + angle

           # angle = angle % 90
                
            # 7. Visualization
            # Draw the rotated box in Green
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
            
            # Draw the angle text
            cv2.putText(frame, f"Angle: {round(angle_deg, 2)}", (int(x), int(y) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # Line 2: Area (Below the center)
            if area > 1100:
                cv2.putText(frame, "big block", (int(x), int(y)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 
            else :
                cv2.putText(frame, "small block", (int(x), int(y)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 
            
            # cv2.putText(frame, f"pixel: {int(x)}, {int(y)}", (int(x), int(y) + 70), 
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) 
        

            return frame, mask, angle_deg
    return None, None, None

# # --- Main Loop (for Webcam) ---
# cap = cv2.VideoCapture(0)

# while True:
#    # ret, frame = cap.read()
#     #if not ret: break
#     image = cv2.imread('test3.png')
#     if image is None:
#         print("Error: could not find image")
    
#     processed_frame, mask = detect_uniqueColors(image, 'orange')
#     # found_color = find_block(image, 875, 359)
#     # processed_frame, mask = detect_uniqueColors(image, found_color)
#     # print(f"found color: {found_color}")


    
#     cv2.imshow('Block Orientation', processed_frame)
#     cv2.imshow('color mask', mask)

    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# #cap.release()
# cv2.destroyAllWindows()