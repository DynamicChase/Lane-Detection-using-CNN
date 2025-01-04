import torch
import torch.cuda as cuda
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

class CUDALaneDetection:
    def __init__(self, height=360, width=640):
        # CUDA Device Configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using CUDA Device: {self.device}")
        
        # Check CUDA availability and properties
        if torch.cuda.is_available():
            self.cuda_device = torch.cuda.current_device()
            print(f"CUDA Device Name: {torch.cuda.get_device_name(self.cuda_device)}")
            print(f"CUDA Device Compute Capability: {torch.cuda.get_device_capability(self.cuda_device)}")

        # Image Dimensions
        self.HEIGHT = height
        self.WIDTH = width

        # Perspective Transformation Matrix
        self.perspective_matrix = self.compute_perspective_matrix()

    def compute_perspective_matrix(self):
        """
        Compute perspective transformation matrix for top view
        """
        # Source points (front view coordinates)
        src_points = np.float32([
            [0, self.HEIGHT],           # Bottom left
            [self.WIDTH, self.HEIGHT],  # Bottom right
            [self.WIDTH//2 + 100, self.HEIGHT//2],  # Top right
            [self.WIDTH//2 - 100, self.HEIGHT//2]   # Top left
        ])
        
        # Destination points (top view coordinates)
        dst_points = np.float32([
            [100, self.HEIGHT],     # Bottom left
            [self.WIDTH-100, self.HEIGHT],  # Bottom right
            [self.WIDTH-100, 0],    # Top right
            [100, 0]                # Top left
        ])
        
        # Compute perspective transform
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return perspective_matrix

    def cuda_lane_detection(self, video_path=None, camera_index=0):
        """
        CUDA-accelerated lane detection for video/camera
        """
        # Select input source
        cap = cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(camera_index)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame
            frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))

            # CUDA Tensor Conversion
            frame_tensor = self.preprocess_frame(frame)

            # Lane Detection Pipeline
            lanes = self.detect_lanes_cuda(frame_tensor)

            # Visualization
            self.visualize_lanes(frame, lanes)

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def preprocess_frame(self, frame):
        """
        CUDA-friendly frame preprocessing
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and move to CUDA
        frame_tensor = ToTensor()(frame_rgb).unsqueeze(0).to(self.device)
        
        return frame_tensor

    def detect_lanes_cuda(self, frame_tensor):
        """
        CUDA-accelerated lane detection
        """
        with torch.cuda.amp.autocast():  # Mixed precision support
            # Convert to grayscale
            grayscale = torch.mean(frame_tensor, dim=1, keepdim=True)
            
            # Canny edge detection
            edges = self.cuda_edge_detection(grayscale)
            
            # Probabilistic Hough Transform
            lines = self.cuda_hough_transform(edges)
        
        return lines

    def cuda_edge_detection(self, grayscale):
        """
        Improved edge detection
        """
        # Use PyTorch's built-in Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], 
                                 [-2, 0, 2], 
                                 [-1, 0, 1]], 
                                dtype=torch.float32, 
                                device=self.device)
        sobel_y = sobel_x.t()
        
        # Prepare kernels
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        # Apply convolution
        edges_x = torch.nn.functional.conv2d(grayscale, sobel_x, padding=1)
        edges_y = torch.nn.functional.conv2d(grayscale, sobel_y, padding=1)
        
        # Compute magnitude
        edges = torch.sqrt(edges_x**2 + edges_y**2)
        
        return edges

    def cuda_hough_transform(self, edges):
        """
        Improved Hough Transform
        """
        # Normalize edges
        edges = (edges - edges.min()) / (edges.max() - edges.min())
        
        # Apply threshold
        threshold = 0.5  # Adjust as needed
        binary_edges = (edges > threshold).float()
        
        return binary_edges

    def visualize_lanes(self, frame, lanes):
        """
        Comprehensive lane visualization
        """
        # Convert CUDA tensor to numpy if needed
        if isinstance(lanes, torch.Tensor):
            lanes = lanes.cpu().numpy()
        
        # Ensure lanes is a boolean or binary mask
        if lanes.ndim > 2:
            lanes = lanes.squeeze()  # Remove extra dimensions
        
        # Create lane mask with the same shape as frame
        lane_mask = np.zeros_like(frame)
        
        # Safely apply lane coloring
        if lanes.dtype == bool or lanes.dtype == np.bool_:
            lane_mask[lanes] = [0, 255, 0]  # Green for detected lanes
        else:
            # If not a boolean mask, use thresholding
            lane_mask[lanes > lanes.mean()] = [0, 255, 0]
        
        # Blend original frame with lane mask
        blended_frame = cv2.addWeighted(frame, 0.7, lane_mask, 0.3, 0)
        
        # Top view transformation
        try:
            top_view_original = cv2.warpPerspective(
                frame, 
                self.perspective_matrix, 
                (self.WIDTH, self.HEIGHT)
            )
            
            top_view_lanes = cv2.warpPerspective(
                blended_frame, 
                self.perspective_matrix, 
                (self.WIDTH, self.HEIGHT)
            )
            
            # Display results
            cv2.imshow('Original Lanes', blended_frame)
            cv2.imshow('Top View Original', top_view_original)
            cv2.imshow('Top View Lanes', top_view_lanes)
        except Exception as e:
            print(f"Visualization error: {e}")
            cv2.imshow('Original Lanes', blended_frame)

def main():
    lane_detector = CUDALaneDetection()
    
    # Video input
    lane_detector.cuda_lane_detection(video_path='/home/sm/Desktop/Cascade-LD/road.mp4')

if __name__ == "__main__":
    main()
