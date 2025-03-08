import cv2
import numpy as np

# Define the points as a NumPy array of type float32
PERSPECTIVE_SRC = np.float32([
    [166, 276],
    [265, 271],
    [414, 309],
    [96, 336]
])

# Create a blank image (for example, 500x500 pixels with 3 channels for color)
img = np.zeros((416, 416, 3), dtype=np.uint8)

# Loop over each point and draw a small circle on the image
for point in PERSPECTIVE_SRC:
    # Convert the float point to integer coordinates for drawing
    pt = tuple(point.astype(int))
    cv2.circle(img, pt, radius=5, color=(0, 255, 0), thickness=-1)

# Optionally, print the coordinates to the console
print("Points drawn on the image:")
print(PERSPECTIVE_SRC)

# Display the image with the drawn points
cv2.imshow("Points", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
