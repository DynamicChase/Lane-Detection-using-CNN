import numpy as np
import cv2
import matplotlib.pyplot as plt

class PreciseTopViewCalibration:
    def __init__(self, width=416, height=416):
        self.width = width
        self.height = height

    def compute_perspective_matrix(self, source_points):
        """
        Compute precise perspective transformation matrix
        """
        # Dynamically compute destination points
        dst_width = self.width
        dst_height = self.height

        destination_points = np.float32([
            [0, dst_height],           # Bottom Left
            [dst_width, dst_height],   # Bottom Right
            [dst_width, 0],            # Top Right
            [0, 0]                     # Top Left
        ])

        # Compute perspective transform
        perspective_matrix = cv2.getPerspectiveTransform(
            np.float32(source_points), 
            destination_points
        )
        
        return perspective_matrix

    def identify_point_types(self, points):
        """
        Identify and classify points based on their coordinates
        """
        # Sort points by y-coordinate (vertical position)
        sorted_points = sorted(points, key=lambda p: p[1])
        
        # Separate top and bottom points
        top_points = sorted_points[:2]
        bottom_points = sorted_points[2:]
        
        # Sort top points by x-coordinate
        top_points = sorted(top_points, key=lambda p: p[0])
        top_left, top_right = top_points
        
        # Sort bottom points by x-coordinate
        bottom_points = sorted(bottom_points, key=lambda p: p[0])
        bottom_left, bottom_right = bottom_points
        
        # Print point classifications
        print("\n--- Point Identification ---")
        print(f"Top Left Point:     {top_left}")
        print(f"Top Right Point:    {top_right}")
        print(f"Bottom Left Point:  {bottom_left}")
        print(f"Bottom Right Point: {bottom_right}")
        
        # Return points in specific order for perspective transform
        return [bottom_left, bottom_right, top_right, top_left]

    def interactive_point_selection(self, image_path):
        """
        Interactive point selection with visualization
        """
        # Read image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.width, self.height))
        
        # Clone for drawing
        display_image = image.copy()
        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Limit to 4 points
                if len(points) < 4:
                    # Draw point
                    cv2.circle(display_image, (x, y), 8, (0, 255, 0), -1)
                    cv2.putText(display_image, str(len(points)+1), (x+10, y+10), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                    points.append([x, y])
                    
                    # Update display
                    cv2.imshow('Point Selection', display_image)
                    print(f"Point {len(points)} Selected: [{x}, {y}]")

        # Create window
        cv2.namedWindow('Point Selection')
        cv2.setMouseCallback('Point Selection', mouse_callback)

        while True:
            cv2.imshow('Point Selection', display_image)
            key = cv2.waitKey(1) & 0xFF

            # Confirm points
            if key == ord('c') and len(points) == 4:
                # Identify and reorder points
                points = self.identify_point_types(points)
                break
            
            # Reset points
            elif key == ord('r'):
                points = []
                display_image = image.copy()
            
            # Quit
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()
        return points

    def apply_perspective_transform(self, image, source_points):
        """
        Apply perspective transformation
        """
        # Compute perspective matrix
        perspective_matrix = self.compute_perspective_matrix(source_points)
        
        # Apply transformation
        warped_image = cv2.warpPerspective(
            image, 
            perspective_matrix, 
            (self.width, self.height),
            flags=cv2.INTER_LINEAR
        )
        
        return warped_image

    def calibrate_top_view(self, image_path):
        """
        Comprehensive top view calibration
        """
        # Read original image
        original_image = cv2.imread(image_path)
        original_image = cv2.resize(original_image, (self.width, self.height))

        # Interactive point selection
        source_points = self.interactive_point_selection(image_path)
        
        if source_points is None:
            return

        # Apply perspective transform
        top_view_image = self.apply_perspective_transform(original_image, source_points)

        # Visualization
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(132)
        plt.title("Selected Points")
        point_image = original_image.copy()
        for i, point in enumerate(source_points, 1):
            cv2.circle(point_image, tuple(point), 8, (0, 255, 0), -1)
            cv2.putText(point_image, str(i), 
                        (point[0]+10, point[1]+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        plt.imshow(cv2.cvtColor(point_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(133)
        plt.title("Top View Transformation")
        plt.imshow(cv2.cvtColor(top_view_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.tight_layout()
        plt.show()

def main():
    calibration = PreciseTopViewCalibration()
    calibration.calibrate_top_view('images/frame.jpg')

if __name__ == "__main__":
    main()