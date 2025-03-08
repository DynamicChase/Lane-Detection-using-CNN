import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
from models.erfnet import Net as ERFNet
import csv
import time
from datetime import datetime

#########################
# CONFIGURATION SECTION #
#########################

DEBUG = False                        # Enable debug output if needed
FRAMEDROP = 4                        # Process every Nth frame in video
IMAGE_HEIGHT = 416                   # Height for resizing images/video frames
IMAGE_WIDTH = 416                    # Width for resizing images/video frames
NUM_CLASSES_SEGMENTATION = 5         # Background + lanes segmentation classes
MODEL_PATH = 'pretrained/erfnet_tusimple.pth'  # Path to the pre-trained model

# Perspective transformation parameters (source points)
PERSPECTIVE_SRC = np.float32([[166, 276],
                              [265, 271],
                              [414, 309],
                              [96, 336]])

######################
# END CONFIG SECTION #
######################

class TopViewLaneDetection:
    def __init__(self, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
        self.HEIGHT = height
        self.WIDTH = width
        self.NUM_CLASSES_SEGMENTATION = NUM_CLASSES_SEGMENTATION  # background + lanes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if DEBUG:
            print(f"[DEBUG] Using Device: {self.device}")
        self.perspective_matrix = self.compute_perspective_matrix()
        self.segmentation_network = self.load_segmentation_network()

    def compute_perspective_matrix(self):
        dst = np.float32([[0, 0], [self.WIDTH, 0], [self.WIDTH, self.HEIGHT], [0, self.HEIGHT]])
        matrix = cv2.getPerspectiveTransform(PERSPECTIVE_SRC, dst)
        if DEBUG:
            print(f"[DEBUG] Computed perspective matrix:\n{matrix}")
        return matrix

    def load_segmentation_network(self):
        network = ERFNet(self.NUM_CLASSES_SEGMENTATION).to(self.device)
        network.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        if DEBUG:
            print(f"[DEBUG] Loaded segmentation network from: {MODEL_PATH}")
        return network.eval()

    def apply_top_view_transform(self, image):
        if isinstance(image, torch.Tensor):
            image = image.squeeze().permute(1, 2, 0).cpu().numpy()
        elif isinstance(image, Image.Image):
            image = np.array(image)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        return cv2.warpPerspective(image, self.perspective_matrix, (self.WIDTH, self.HEIGHT), flags=cv2.INTER_LINEAR)

    def transform_to_top_view(self, lane_pixels):
        if len(lane_pixels) == 0:
            return []
        # Convert (row, col) -> (x, y)
        points = lane_pixels[:, [1, 0]].astype(np.float32)
        points = points.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, self.perspective_matrix)
        return np.round(transformed.reshape(-1, 2)).astype(int)

    def compress_lane_points(self, lane_points):
        from collections import defaultdict
        if len(lane_points) == 0:
            return []
        row_dict = defaultdict(list)
        for x, y in lane_points:
            row_dict[y].append(x)
        compressed = []
        for y in sorted(row_dict.keys()):
            med_x = int(round(np.median(row_dict[y])))
            compressed.append([med_x, y])
        return np.array(compressed, dtype=int)

    def fit_lane_to_straight_line(self, lane_points, deg=1):
        if len(lane_points) < 2:
            return lane_points
        lane_points = lane_points[np.argsort(lane_points[:, 1])]
        x_vals = lane_points[:, 0]
        y_vals = lane_points[:, 1]
        poly = np.polyfit(y_vals, x_vals, deg=deg)
        y_new = np.arange(y_vals.min(), y_vals.max() + 1, 1)
        x_new = np.polyval(poly, y_new)
        return np.stack([np.round(x_new).astype(int), y_new], axis=1)

    def group_lane_segments(self, segments, threshold):
        groups = []
        if not segments:
            return groups
        seg_centroids = [(np.mean(seg[:, 0]), seg) for seg in segments if seg.size]
        seg_centroids.sort(key=lambda x: x[0])
        current = [seg_centroids[0]]
        for item in seg_centroids[1:]:
            if abs(item[0] - current[-1][0]) < threshold:
                current.append(item)
            else:
                groups.append(current)
                current = [item]
        if current:
            groups.append(current)
        merged = []
        for group in groups:
            pts = np.vstack([seg for (_, seg) in group])
            comp = self.compress_lane_points(pts)
            fit = self.fit_lane_to_straight_line(comp, deg=1)
            merged.append((np.mean(fit[:, 0]), fit))
        merged.sort(key=lambda x: x[0])
        return merged

    def extract_lane_coordinates(self, segmentation_mask):
        mask_np = segmentation_mask.cpu().numpy()[0]
        unique = np.unique(mask_np)
        unique = unique[unique != 0]
        segments = []
        kernel = np.ones((5, 5), np.uint8)
        for label in unique:
            binary = ((mask_np == label).astype(np.uint8)) * 255
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            pixels = np.column_stack(np.where(cleaned > 0))
            if pixels.shape[0] == 0:
                continue
            pixels = pixels[np.argsort(pixels[:, 0])]
            top = self.transform_to_top_view(pixels)
            comp = self.compress_lane_points(top)
            fit = self.fit_lane_to_straight_line(comp, deg=1)
            segments.append(fit)
        if len(segments) == 0:
            return {"left_lane": None, "separation_lanes": [], "right_lane": None}
        if len(segments) == 1:
            cent = np.mean(segments[0][:, 0])
            if cent < self.WIDTH / 2:
                return {"left_lane": segments[0], "separation_lanes": [], "right_lane": None}
            else:
                return {"left_lane": None, "separation_lanes": [], "right_lane": segments[0]}
        groups = self.group_lane_segments(segments, threshold=self.WIDTH * 0.1)
        out = {"left_lane": None, "separation_lanes": [], "right_lane": None}
        if len(groups) == 1:
            cent, line = groups[0]
            if cent < self.WIDTH / 2:
                out["left_lane"] = line
            else:
                out["right_lane"] = line
        else:
            out["left_lane"] = groups[0][1]
            out["right_lane"] = groups[-1][1]
            if len(groups) > 2:
                for grp in groups[1:-1]:
                    out["separation_lanes"].append(grp[1])
        if DEBUG:
            print(f"[DEBUG] Extracted lane coordinates: {out}")
        return out

    def save_lanes_to_csv(self, lanes, output_csv_path, frame=None):
        mode = 'a' if frame is not None else 'w'
        header = ['frame', 'lane_type', 'x', 'y'] if frame is not None else ['lane_type', 'x', 'y']
        with open(output_csv_path, mode, newline='') as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                writer.writerow(header)
            if lanes["left_lane"] is not None:
                for point in lanes["left_lane"]:
                    if frame is not None:
                        writer.writerow([frame, 'left_lane', point[0], point[1]])
                    else:
                        writer.writerow(['left_lane', point[0], point[1]])
            if lanes["right_lane"] is not None:
                for point in lanes["right_lane"]:
                    if frame is not None:
                        writer.writerow([frame, 'right_lane', point[0], point[1]])
                    else:
                        writer.writerow(['right_lane', point[0], point[1]])
            for sep_lane in lanes["separation_lanes"]:
                for point in sep_lane:
                    if frame is not None:
                        writer.writerow([frame, 'separation_lane', point[0], point[1]])
                    else:
                        writer.writerow(['separation_lane', point[0], point[1]])
        if DEBUG:
            if frame is not None:
                print(f"[DEBUG] Saved lane coordinates for frame {frame} to {output_csv_path}")
            else:
                print(f"[DEBUG] Saved lane coordinates to {output_csv_path}")

    def construct_lane_boundaries(self, lanes_dict):
        boundaries = []
        if lanes_dict["left_lane"] is not None:
            boundaries.append(lanes_dict["left_lane"])
        boundaries.extend(lanes_dict["separation_lanes"])
        if lanes_dict["right_lane"] is not None:
            boundaries.append(lanes_dict["right_lane"])
        return boundaries

    def construct_two_lanes(self, lanes_dict):
        boundaries = self.construct_lane_boundaries(lanes_dict)
        if len(boundaries) < 3:
            if DEBUG:
                print("[DEBUG] Not enough boundaries to construct 2 lanes.")
            return None
        lane1 = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        lane2 = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        pts_b1 = boundaries[0].reshape((-1, 1, 2))
        pts_b2 = boundaries[1].reshape((-1, 1, 2))
        pts_b3 = boundaries[2].reshape((-1, 1, 2))
        cv2.polylines(lane1, [pts_b1], isClosed=False, color=(0, 0, 255), thickness=2)
        cv2.polylines(lane1, [pts_b2], isClosed=False, color=(255, 0, 0), thickness=2)
        cv2.polylines(lane2, [pts_b2], isClosed=False, color=(0, 0, 255), thickness=2)
        cv2.polylines(lane2, [pts_b3], isClosed=False, color=(255, 0, 0), thickness=2)
        if DEBUG:
            print("[DEBUG] Constructed two lanes from boundaries.")
        return lane1, lane2

    def visualize_lane_detection(self, original, seg_mask, lanes_dict):
        mask_np = seg_mask.cpu().numpy()[0]
        color_mask = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        for label in np.unique(mask_np):
            if label == 0:
                continue
            color = np.random.randint(0, 255, 3).tolist()
            color_mask[mask_np == label] = color

        top_view = self.apply_top_view_transform(original)
        top_mask = self.apply_top_view_transform(color_mask)
        lines = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        if lanes_dict["left_lane"] is not None:
            pts = lanes_dict["left_lane"].reshape((-1, 1, 2))
            cv2.polylines(lines, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
        if lanes_dict["right_lane"] is not None:
            pts = lanes_dict["right_lane"].reshape((-1, 1, 2))
            cv2.polylines(lines, [pts], isClosed=False, color=(255, 0, 0), thickness=2)
        for sep in lanes_dict["separation_lanes"]:
            pts = sep.reshape((-1, 1, 2))
            cv2.polylines(lines, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

        plt.figure(figsize=(18, 10))
        plt.subplot(2, 3, 1)
        plt.title("Original Image")
        plt.imshow(np.array(original))
        plt.axis('off')
        plt.subplot(2, 3, 2)
        plt.title("Lane Segmentation Mask")
        plt.imshow(color_mask)
        plt.axis('off')
        plt.subplot(2, 3, 3)
        plt.title("Top View Original")
        plt.imshow(cv2.cvtColor(top_view, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.subplot(2, 3, 4)
        plt.title("Top View Mask")
        plt.imshow(cv2.cvtColor(top_mask, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.subplot(2, 3, 5)
        plt.title("Detected Lane Boundaries")
        plt.imshow(cv2.cvtColor(lines, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        two_lanes = self.construct_two_lanes(lanes_dict)
        if two_lanes is None:
            print("Not enough boundaries to form 2 lanes with the desired pairing (need at least 3 boundaries).")
            return
        lane1, lane2 = two_lanes
        cv2.imshow("Lane 1 (Boundary 1 & 2)", lane1)
        cv2.imshow("Lane 2 (Boundary 2 & 3)", lane2)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Lane 1 (Boundary 1 & 2)")
        plt.imshow(cv2.cvtColor(lane1, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title("Lane 2 (Boundary 2 & 3)")
        plt.imshow(cv2.cvtColor(lane2, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def visualize_lane_detection_video(self, original, seg_mask, frame, lanes_dict):
        mask_np = seg_mask.cpu().numpy()[0]
        color_mask = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        for label in np.unique(mask_np):
            if label == 0:
                continue
            color = np.random.randint(0, 255, 3).tolist()
            color_mask[mask_np == label] = color

        top_view = self.apply_top_view_transform(original)
        top_mask = self.apply_top_view_transform(color_mask)
        lines = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        if lanes_dict["left_lane"] is not None:
            pts = lanes_dict["left_lane"].reshape((-1, 1, 2))
            cv2.polylines(lines, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
        if lanes_dict["right_lane"] is not None:
            pts = lanes_dict["right_lane"].reshape((-1, 1, 2))
            cv2.polylines(lines, [pts], isClosed=False, color=(255, 0, 0), thickness=2)
        for sep in lanes_dict["separation_lanes"]:
            pts = sep.reshape((-1, 1, 2))
            cv2.polylines(lines, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

        blended = cv2.addWeighted(cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR),
                                  0.7, color_mask, 0.3, 0)
        cv2.imshow('Original Lane Detection', blended)
        cv2.imshow('Top View Original', top_view)
        cv2.imshow('Top View Mask', top_mask)
        cv2.imshow('Detected Lane Boundaries', lines)
        
        two_lanes = self.construct_two_lanes(lanes_dict)
        if two_lanes is not None:
            lane1, lane2 = two_lanes
            cv2.imshow("Lane 1 (Boundary 1 & 2)", lane1)
            cv2.imshow("Lane 2 (Boundary 2 & 3)", lane2)

    def compute_lane_midpoint(self, lane_points):
        """Fallback: Compute the midpoint of the lane points."""
        if lane_points is None or len(lane_points) == 0:
            return None
        mid_index = len(lane_points) // 2
        return tuple(lane_points[mid_index])

    def get_start_and_goal_points(self, lanes_dict, obstacle_in_lane1=False):
        """
        Compute the static start point (bottom-center) and two goal points based only on the 
        topmost data from the left and right lanes.
        
        - If both left and right lane data exist:
            • Find the topmost point (smallest y) in the left lane and right lane.
            • If a separation lane is available, compute:
                Goal 1 = midpoint of left lane topmost and separation lane topmost.
                Goal 2 = midpoint of separation lane topmost and right lane topmost.
            • Otherwise, both goals are set to the midpoint between left and right lane topmost.
        - If only one lane is detected, use its topmost point.
        """
        start_point = (self.WIDTH // 2, self.HEIGHT)  # e.g., (208, 416)
        left_lane = lanes_dict.get("left_lane")
        right_lane = lanes_dict.get("right_lane")
        sep_lanes = lanes_dict.get("separation_lanes", [])
        
        if left_lane is not None and right_lane is not None:
            # Get topmost (smallest y) from left and right lane data
            left_top = left_lane[np.argmin(left_lane[:, 1])]
            right_top = right_lane[np.argmin(right_lane[:, 1])]
            if len(sep_lanes) > 0:
                # Use first separation lane's topmost point if available
                sep = sep_lanes[0]
                sep_top = sep[np.argmin(sep[:, 1])]
                goal1 = ((left_top[0] + sep_top[0]) // 2, (left_top[1] + sep_top[1]) // 2)
                goal2 = ((sep_top[0] + right_top[0]) // 2, (sep_top[1] + right_top[1]) // 2)
            else:
                # If no separation lane, use the midpoint between left and right
                mid_goal = ((left_top[0] + right_top[0]) // 2, (left_top[1] + right_top[1]) // 2)
                goal1 = mid_goal
                goal2 = mid_goal
        elif left_lane is not None:
            left_top = left_lane[np.argmin(left_lane[:, 1])]
            goal1 = left_top
            goal2 = left_top
        elif right_lane is not None:
            right_top = right_lane[np.argmin(right_lane[:, 1])]
            goal1 = right_top
            goal2 = right_top
        else:
            goal1 = (self.WIDTH // 2, self.HEIGHT // 2)
            goal2 = (self.WIDTH // 2, self.HEIGHT // 2)
        
        print("Static Start point (x,y):", start_point)
        print("Goal point 1 (x,y):", goal1)
        print("Goal point 2 (x,y):", goal2)
        return start_point, goal1, goal2

    def detect_lanes(self, image_path, csv_path=None):
        if csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"lane_coordinates_{timestamp}.csv"
            if DEBUG:
                print(f"[DEBUG] Generated CSV filename: {csv_path}")
        original = Image.open(image_path).resize((self.WIDTH, self.HEIGHT))
        tensor = ToTensor()(original).unsqueeze(0).to(self.device)
        with torch.no_grad():
            seg_out = self.segmentation_network(tensor)
            seg_mask = seg_out.max(dim=1)[1]
        lanes = self.extract_lane_coordinates(seg_mask)
        self.save_lanes_to_csv(lanes, csv_path)
        self.visualize_lane_detection(original, seg_mask, lanes)
        # For demonstration, set obstacle flag manually (replace with your actual logic)
        obstacle_in_lane1 = False  
        start_point, goal1, goal2 = self.get_start_and_goal_points(lanes, obstacle_in_lane1)
        # Overlay these points on the top-view image for visualization:
        top_view = self.apply_top_view_transform(original)
        cv2.circle(top_view, start_point, 5, (0, 255, 0), -1)  # start in green
        if goal1 is not None:
            cv2.circle(top_view, goal1, 5, (0, 0, 255), -1)   # goal 1 in red
        if goal2 is not None:
            cv2.circle(top_view, goal2, 5, (255, 0, 0), -1)   # goal 2 in blue
        plt.figure(figsize=(8, 8))
        plt.title("Start and Goal Points on Top-View")
        plt.imshow(cv2.cvtColor(top_view, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def detect_lanes_video(self, video_path=None, camera_index=0, csv_path=None):
        if csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"video_lane_coordinates_{timestamp}.csv"
            if DEBUG:
                print(f"[DEBUG] Generated CSV filename: {csv_path}")
        cap = cv2.VideoCapture(video_path) if video_path else cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % FRAMEDROP != 0:
                continue
            if DEBUG:
                print(f"[DEBUG] Processing frame {frame_count}")
            frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
            original = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor = ToTensor()(original).unsqueeze(0).to(self.device)
            with torch.no_grad():
                seg_out = self.segmentation_network(tensor)
                seg_mask = seg_out.max(dim=1)[1]
            lanes = self.extract_lane_coordinates(seg_mask)
            self.save_lanes_to_csv(lanes, csv_path, frame=frame_count)
            self.visualize_lane_detection_video(original, seg_mask, frame_count, lanes)
            # For demonstration, set obstacle flag manually (replace with your actual logic)
            obstacle_in_lane1 = False  
            start_point, goal1, goal2 = self.get_start_and_goal_points(lanes, obstacle_in_lane1)
            top_view = self.apply_top_view_transform(original)
            cv2.circle(top_view, start_point, 5, (0, 255, 0), -1)
            if goal1 is not None:
                cv2.circle(top_view, goal1, 5, (0, 0, 255), -1)
            if goal2 is not None:
                cv2.circle(top_view, goal2, 5, (255, 0, 0), -1)
            cv2.imshow("Start & Goal Points on Top-View", top_view)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

def main():
    lane_detector = TopViewLaneDetection()
    # For image detection (with CSV saving and start/goal point determination):
    # Replace '/path/to/your/image.jpg' with your actual image file.
    # lane_detector.detect_lanes('/path/to/your/image.jpg')
    
    # For video detection:
    lane_detector.detect_lanes_video(video_path='try.mp4')
    time.sleep(0.1)
    # Or use a live camera feed:
    # lane_detector.detect_lanes_video(camera_index=0)

if __name__ == "__main__":
    main()
