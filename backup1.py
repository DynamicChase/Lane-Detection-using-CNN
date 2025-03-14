import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Image Processing
from PIL import Image
from torchvision.transforms import ToTensor

# Custom Modules (Ensure these exist)
from models.erfnet import Net as ERFNet
from models.lcnet import Net as LCNet
from functions import color_lanes, blend

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
        Using specific points for 416x416 resolution
        """
        # Source points (front view coordinates)
        src_points = np.float32([
            [57, 399],    # Bottom Left Point
            [410, 395],   # Bottom Right Point
            [378, 189],   # Top Right Point
            [57, 182]     # Top Left Point
        ])
        
        # Destination points (top view coordinates)
        dst_points = np.float32([
            [50, 416],        # Bottom Left
            [366, 416],       # Bottom Right
            [366, 0],         # Top Right
            [50, 0]           # Top Left
        ])
        
        # Compute perspective transform
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return perspective_matrix

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
        """
        Convert image to top view perspective
        """
        # Convert input to numpy array
        if isinstance(image, torch.Tensor):
            image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply perspective transformation
        top_view = cv2.warpPerspective(
            image, 
            self.perspective_matrix, 
            (self.WIDTH, self.HEIGHT), 
            flags=cv2.INTER_LINEAR
        )
        
        return top_view

    def detect_lanes(self, image_path):
        """
        Comprehensive lane detection pipeline
        """
        try:
            # Load and preprocess image
            original_image = Image.open(image_path)
            original_image = original_image.resize((self.WIDTH, self.HEIGHT))
            
            # Convert to tensor
            image_tensor = ToTensor()(original_image).unsqueeze(0).to(self.device)
            
            # Perform segmentation
            with torch.no_grad():
                segmentation_output = self.segmentation_network(image_tensor)
                segmentation_mask = segmentation_output.max(dim=1)[1]
            
            # Visualize Results
            self.visualize_lane_detection(
                original_image, 
                segmentation_mask
            )
        
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
        except Exception as e:
            print(f"Error in lane detection: {e}")

    def detect_lanes_video(self, video_path=None, camera_index=0):
        """
        Lane detection for video or camera stream
        """
        # Select input source
        cap = cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(camera_index)

        # Set video capture resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to 416x416
            frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
            
            # Convert frame to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Convert to tensor
            image_tensor = ToTensor()(pil_image).unsqueeze(0).to(self.device)
            
            # Perform segmentation
            with torch.no_grad():
                segmentation_output = self.segmentation_network(image_tensor)
                segmentation_mask = segmentation_output.max(dim=1)[1]
            
            # Visualize Results in Real-time
            self.visualize_lane_detection_video(
                pil_image, 
                segmentation_mask,
                frame
            )
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def visualize_lane_detection(self, original_image, segmentation_mask):
        """
        Create comprehensive visualization
        """
        # Convert mask to numpy
        mask_np = segmentation_mask.cpu().numpy()[0]
        
        # Create color mask
        color_mask = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        for lane_id in range(1, self.NUM_CLASSES_SEGMENTATION):
            color = np.random.randint(0, 255, 3).tolist()
            color_mask[mask_np == lane_id] = color
        
        # Apply top view transformations
        original_top_view = self.apply_top_view_transform(original_image)
        mask_top_view = self.apply_top_view_transform(color_mask)
        
        # Create visualization plot
        plt.figure(figsize=(20, 10))
        
        # Original Image
        plt.subplot(2, 2, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR))
        plt.axis('off')
        
        # Segmentation Mask
        plt.subplot(2, 2, 2)
        plt.title("Lane Segmentation Mask")
        plt.imshow(color_mask)
        plt.axis('off')
        
        # Top View Original
        plt.subplot(2, 2, 3)
        plt.title("Top View Original")
        plt.imshow(cv2.cvtColor(original_top_view, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # Top View Lane Mask
        plt.subplot(2, 2, 4)
        plt.title("Top View Lane Mask")
        plt.imshow(cv2.cvtColor(mask_top_view, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def visualize_lane_detection_video(self, original_image, segmentation_mask, original_frame):
        """
        Real-time lane detection visualization
        """
        # Convert mask to numpy
        mask_np = segmentation_mask.cpu().numpy()[0]
        
        # Create color mask
        color_mask = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        for lane_id in range(1, self.NUM_CLASSES_SEGMENTATION):
            color = np.random.randint(0, 255, 3).tolist()
            color_mask[mask_np == lane_id] = color
        
        # Apply top view transformations
        original_top_view = self.apply_top_view_transform(original_image)
        mask_top_view = self.apply_top_view_transform(color_mask)
        
        # Blend original frame with lane mask
        blended_frame = cv2.addWeighted(
            cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR), 
            0.7, 
            color_mask, 
            0.3, 
            0
        )
        
        # Display multiple windows
        cv2.imshow('Original Lane Detection', blended_frame)
        cv2.imshow('Top View Original', original_top_view)
        cv2.imshow('Top View Lane Mask', mask_top_view)

def main():
    lane_detector = TopViewLaneDetection()
    
    # Options:
    # 1. Image Detection
    # lane_detector.detect_lanes('/home/sm/Desktop/Cascade-LD/images/test.jpg')
    
    # 2. Video Detection
    lane_detector.detect_lanes_video(video_path='/home/sm/Desktop/Cascade-LD/road.mp4')
    
    # 3. Live Camera Detection
    # lane_detector.detect_lanes_video()

if __name__ == "__main__":
    main()
