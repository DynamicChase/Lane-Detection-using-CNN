import cv2
import numpy as np
import matplotlib.pyplot as plt

def perspective_transform(image, src_points):
    """
    Perform perspective transformation to get top view
    """
    # Destination points for 416x416 frame
    dst_points = np.float32([
        [2, 320],         # Bottom-left
        [414, 320],       # Bottom-right
        [270, 264],         # Top-right
        [141, 264]            # Top-left
    ])

    # Compute perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply perspective transformation with interpolation
    top_view = cv2.warpPerspective(image, matrix, (416, 416), 
                                   flags=cv2.INTER_LINEAR, 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(0, 0, 0))
    
    return top_view

# Load image
image = cv2.imread('/home/sm/Desktop/road/images/campus.jpg')
image = cv2.resize(image, (416, 416))

# Calibration points (carefully adjusted)
src_points = np.float32([
     [45, 414],   # Bottom-left
        [356, 410],   # Bottom-right
        [262, 273],   # Top-right
        [156, 270]    # Top-left
])

# Generate top view
top_view = perspective_transform(image, src_points)

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(132)
plt.title('Calibration Points')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
for (x, y) in src_points:
    plt.scatter(x, y, color='red', s=50)

plt.subplot(133)
plt.title('Top View')
plt.imshow(cv2.cvtColor(top_view, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()
