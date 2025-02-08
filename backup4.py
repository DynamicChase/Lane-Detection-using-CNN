import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor

# Custom Modules
from models.erfnet import Net as ERFNet
from models.lcnet import Net as LCNet

class TopViewLaneDetection:
    def __init__(self, height=416, width=416):
        # Image Dimensions
        self.HEIGHT = height
        self.WIDTH = width

        # Network Configuration
        self.DESCRIPTOR_SIZE = 64
        self.NUM_CLASSES_SEGMENTATION = 5
        self.NUM_CLASSES_CLASSIFICATION = 3

        # Device Configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using Device: {self.device}")

        # Perspective Transformation Matrix
        self.perspective_matrix = self.compute_perspective_matrix()

        # Initialize Networks
        self.segmentation_network = self.load_segmentation_network()
        self.classification_network = self.load_classification_network()

    def compute_perspective_matrix(self):
        """ 
        Compute perspective transformation matrix for top view 
        Adjust the four points here for your calibration.
        """
        # Source points (front view coordinates) -- update these as per your calibration
        src_points = np.float32([
            [97, 379],   # Bottom Left Point
            [412, 330],  # Bottom Right Point
            [245, 271],  # Top Right Point
            [175, 273]   # Top Left Point
        ])

        # Destination points (top view coordinates) -- update these if needed
        dst_points = np.float32([
            [97, 379],   # Bottom Left 
            [412, 330],  # Bottom Right 
            [245, 271],  # Top Right 
            [175, 273]   # Top Left 
        ])

        # Compute perspective transform
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return perspective_matrix

    def mask_area_below_line(self, frame):
        """
        Masks the area below the line defined by points (0, 250) and (416, 250).
        
        Parameters:
            frame (np.ndarray): input image frame.
        Returns:
            masked_frame (np.ndarray): image with the area below the line masked.
        """
        height, width, _ = frame.shape
        polygon = np.array([[
            (0, 250),
            (width, 250),
            (width, height),
            (0, height)
        ]], dtype=np.int32)
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, polygon, (255, 255, 255))
        masked_frame = cv2.bitwise_and(frame, mask)
        return masked_frame

    def mask_area_above_line(self, frame, line_y=250):
        """
        Masks the upper area of the frame above the specified horizontal line.
        
        Parameters:
            frame (np.ndarray): input image frame.
            line_y (int): y-coordinate of the horizontal line.
        Returns:
            masked_frame (np.ndarray): image with the area above the line masked.
        """
        height, width, _ = frame.shape
        # Define polygon for upper area (from top to the line_y)
        polygon = np.array([[
            (0, 0),
            (width, 0),
            (width, line_y),
            (0, line_y)
        ]], dtype=np.int32)
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, polygon, (255, 255, 255))
        masked_frame = cv2.bitwise_and(frame, mask)
        return masked_frame

    def load_segmentation_network(self):
        """Load pre-trained segmentation network"""
        try:
            network = ERFNet(self.NUM_CLASSES_SEGMENTATION).to(self.device)
            network.load_state_dict(
                torch.load('pretrained/erfnet_tusimple.pth', map_location=self.device)
            )
            return network.eval()
        except FileNotFoundError:
            print("Segmentation model not found!")
            return None
        except Exception as e:
            print(f"Error loading segmentation model: {e}")
            return None

    def load_classification_network(self):
        """Load pre-trained classification network"""
        try:
            model_path = f'pretrained/classification_{self.DESCRIPTOR_SIZE}_{self.NUM_CLASSES_CLASSIFICATION}class.pth'
            network = LCNet(
                self.NUM_CLASSES_CLASSIFICATION, 
                self.DESCRIPTOR_SIZE, 
                self.DESCRIPTOR_SIZE
            ).to(self.device)
            network.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            return network.eval()
        except FileNotFoundError:
            print("Classification model not found!")
            return None
        except Exception as e:
            print(f"Error loading classification model: {e}")
            return None

    def apply_top_view_transform(self, image):
        """ Convert image to top view perspective """
        if isinstance(image, torch.Tensor):
            image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        elif isinstance(image, Image.Image):
            image = np.array(image)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        top_view = cv2.warpPerspective(
            image, 
            self.perspective_matrix, 
            (self.WIDTH, self.HEIGHT), 
            flags=cv2.INTER_LINEAR
        )
        return top_view

    def extract_lane_coordinates(self, segmentation_mask):
        mask_np = segmentation_mask.cpu().numpy()[0]
        lane_coordinates = {
            'left_lane': [],
            'right_lane': []
        }
        for lane_id in range(1, self.NUM_CLASSES_SEGMENTATION):
            lane_pixels = np.column_stack(np.where(mask_np == lane_id))
            lane_pixels = lane_pixels[lane_pixels[:, 0].argsort()]
            if len(lane_pixels) > 0:
                avg_x = np.mean(lane_pixels[:, 1])
                if avg_x < self.WIDTH / 2:
                    lane_coordinates['left_lane'].append(lane_pixels)
                else:
                    lane_coordinates['right_lane'].append(lane_pixels)
        self.save_lane_coordinates(lane_coordinates)
        return lane_coordinates

    def extract_lane_coordinates_from_top_view(self, mask_top_view):
        mask_gray = cv2.cvtColor(mask_top_view, cv2.COLOR_BGR2GRAY)
        _, mask_binary = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
        lane_pixels = np.column_stack(np.where(mask_binary > 0))
        lane_coordinates = {
            'left_lane': [],
            'right_lane': []
        }
        if len(lane_pixels) > 0:
            lane_pixels = lane_pixels[lane_pixels[:, 0].argsort()]
            left_lane_pixels = lane_pixels[lane_pixels[:, 1] < 200]
            right_lane_pixels = lane_pixels[lane_pixels[:, 1] >= 200]
            if len(left_lane_pixels) > 0:
                lane_coordinates['left_lane'].append(left_lane_pixels)
            if len(right_lane_pixels) > 0:
                lane_coordinates['right_lane'].append(right_lane_pixels)
        self.print_lane_coordinates(lane_coordinates)
        return lane_coordinates

    def save_lane_coordinates(self, lane_coordinates):
        try:
            with open('lane_coordinates.txt', 'w') as f:
                f.write("Lane Coordinates\n")
                f.write("================\n\n")
                f.write("Left Lane Coordinates:\n")
                for i, lane in enumerate(lane_coordinates['left_lane'], 1):
                    f.write(f"Left Lane {i}:\n")
                    for point in lane:
                        f.write(f" Y: {point[0]}, X: {point[1]}\n")
                f.write("\nRight Lane Coordinates:\n")
                for i, lane in enumerate(lane_coordinates['right_lane'], 1):
                    f.write(f"Right Lane {i}:\n")
                    for point in lane:
                        f.write(f" Y: {point[0]}, X: {point[1]}\n")
            print("Lane coordinates saved to lane_coordinates.txt")
        except Exception as e:
            print(f"Error saving lane coordinates: {e}")

    def print_lane_coordinates(self, lane_coordinates):
        print("\n--- Lane Coordinates ---")
        print("Left Lane Coordinates:")
        for i, lane in enumerate(lane_coordinates['left_lane'], 1):
            print(f"Left Lane {i}:")
            for point in lane:
                print(f" Y: {point[0]}, X: {point[1]}")
        print("\nRight Lane Coordinates:")
        for i, lane in enumerate(lane_coordinates['right_lane'], 1):
            print(f"Right Lane {i}:")
            for point in lane:
                print(f" Y: {point[0]}, X: {point[1]}")

    def detect_lanes(self, image_path):
        try:
            original_image = Image.open(image_path)
            original_image = original_image.resize((self.WIDTH, self.HEIGHT))
            image_tensor = ToTensor()(original_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                segmentation_output = self.segmentation_network(image_tensor)
                segmentation_mask = segmentation_output.max(dim=1)[1]
            lane_coords = self.extract_lane_coordinates(segmentation_mask)
            self.visualize_lane_detection(original_image, segmentation_mask)
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
        except Exception as e:
            print(f"Error in lane detection: {e}")

    def detect_lanes_video(self, video_path=None, camera_index=0):
        """ Lane detection for video or camera stream """
        cap = cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
            # Create a masked version for upper area (masking the area above y=250)
            upper_masked = self.mask_area_above_line(frame, line_y=250)
            top_view_upper = self.apply_top_view_transform(upper_masked)
            # Convert frame to PIL Image for segmentation processing
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = ToTensor()(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                segmentation_output = self.segmentation_network(image_tensor)
                segmentation_mask = segmentation_output.max(dim=1)[1]
            self.visualize_lane_detection_video(pil_image, segmentation_mask, frame)
            cv2.imshow('Upper Masked Area', upper_masked)
            cv2.imshow('Top View of Upper Masked Area', top_view_upper)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def visualize_lane_detection(self, original_image, segmentation_mask):
        mask_np = segmentation_mask.cpu().numpy()[0]
        color_mask = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        for lane_id in range(1, self.NUM_CLASSES_SEGMENTATION):
            color = np.random.randint(0, 255, 3).tolist()
            color_mask[mask_np == lane_id] = color
        original_top_view = self.apply_top_view_transform(original_image)
        mask_top_view = self.apply_top_view_transform(color_mask)
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR))
        plt.axis('off')
        plt.subplot(2, 2, 2)
        plt.title("Lane Segmentation Mask")
        plt.imshow(color_mask)
        plt.axis('off')
        plt.subplot(2, 2, 3)
        plt.title("Top View Original")
        plt.imshow(cv2.cvtColor(original_top_view, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.subplot(2, 2, 4)
        plt.title("Top View Lane Mask")
        plt.imshow(cv2.cvtColor(mask_top_view, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def visualize_lane_detection_video(self, original_image, segmentation_mask, original_frame):
        mask_np = segmentation_mask.cpu().numpy()[0]
        color_mask = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        for lane_id in range(1, self.NUM_CLASSES_SEGMENTATION):
            color = np.random.randint(0, 255, 3).tolist()
            color_mask[mask_np == lane_id] = color
        original_top_view = self.apply_top_view_transform(original_image)
        mask_top_view = self.apply_top_view_transform(color_mask)
        lane_coordinates = self.extract_lane_coordinates_from_top_view(mask_top_view)
        blended_frame = cv2.addWeighted(
            cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR), 
            0.7, 
            color_mask, 
            0.3, 
            0
        )
        cv2.imshow('Original Lane Detection', blended_frame)
        cv2.imshow('Top View Original', original_top_view)
        cv2.imshow('Top View Lane Mask', mask_top_view)

# Main function for video/camera detection
def main():
    lane_detector = TopViewLaneDetection()
    
    # Option 1: Process a video file (replace 'try.mp4' with your video file path)
    video_path = 'try.mp4'
    lane_detector.detect_lanes_video(video_path=video_path)
    
    # Option 2: Process live camera feed (uncomment the next line to use your camera)
    # lane_detector.detect_lanes_video(camera_index=0)

if __name__ == "__main__":
    main()
