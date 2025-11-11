import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO models
pose_model = YOLO('yolov8n-pose.pt')  # Load YOLOv8 pose model

def calculate_pose_similarity(landmarks1, landmarks2):
    """Calculate similarity between two poses using landmark coordinates."""
    if not landmarks1 or not landmarks2:
        return 0.0
    
    # Convert landmarks to numpy arrays for easier calculation
    points1 = np.array([[lm.x, lm.y, lm.z] for lm in landmarks1])
    points2 = np.array([[lm.x, lm.y, lm.z] for lm in landmarks2])
    
    # Calculate cosine similarity
    dot_product = np.sum(points1 * points2)
    norm1 = np.linalg.norm(points1)
    norm2 = np.linalg.norm(points2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    similarity = dot_product / (norm1 * norm2)
    return (similarity + 1) / 2 * 100  # Convert to percentage

class PoseDetector:
    def __init__(self, similarity_threshold=90.0):
        self.pose_model = pose_model
        self.similarity_threshold = similarity_threshold
        self.current_keypoints = None
        self.highest_similarity = 0.0
        self.best_frame = None

    def normalize_keypoints(self, keypoints):
        """Normalize keypoints to be scale and translation invariant."""
        if keypoints is None or len(keypoints) == 0:
            return None
            
        # Convert to numpy array
        points = np.array(keypoints)
        
        # Calculate center and scale
        center = np.mean(points, axis=0)
        points_centered = points - center
        scale = np.sqrt(np.mean(np.sum(points_centered**2, axis=1)))
        
        if scale > 0:
            points_normalized = points_centered / scale
        else:
            return None
            
        return points_normalized

    def calculate_pose_similarity(self, keypoints1, keypoints2):
        """Calculate similarity between two poses using normalized keypoints."""
        if keypoints1 is None or keypoints2 is None:
            return 0.0
            
        # Define key points with higher weights
        key_points_weights = {
            # Shoulders
            5: 1.5, 6: 1.5,  # YOLOv8 shoulder indices
            # Elbows
            7: 1.2, 8: 1.2,  # YOLOv8 elbow indices
            # Wrists
            9: 1.2, 10: 1.2,  # YOLOv8 wrist indices
            # Hips
            11: 1.5, 12: 1.5,  # YOLOv8 hip indices
            # Knees
            13: 1.2, 14: 1.2,  # YOLOv8 knee indices
            # Ankles
            15: 1.2, 16: 1.2,  # YOLOv8 ankle indices
        }
        
        # Normalize keypoints
        norm_kpts1 = self.normalize_keypoints(keypoints1)
        norm_kpts2 = self.normalize_keypoints(keypoints2)
        
        if norm_kpts1 is None or norm_kpts2 is None:
            return 0.0
        
        total_similarity = 0
        total_weight = 0
        
        # Calculate weighted similarity for each keypoint
        for i in range(len(norm_kpts1)):
            weight = key_points_weights.get(i, 1.0)
            
            point1 = norm_kpts1[i]
            point2 = norm_kpts2[i]
            
            # Calculate Euclidean distance-based similarity
            distance = np.linalg.norm(point1 - point2)
            point_similarity = np.exp(-distance)  # Convert distance to similarity score
            
            total_similarity += point_similarity * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
            
        # Calculate weighted average similarity and convert to regular Python float
        avg_similarity = float(total_similarity / total_weight)
        # Convert to percentage
        return float(avg_similarity * 100)

    def process_frame(self, frame, target_keypoints=None):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO pose detection
        results = self.pose_model(img_rgb, imgsz = 384)[0]
        
        similarity = 0.0
        draw_color = (200, 200, 200)
        status = "No Person"

        if results.keypoints is not None and len(results.keypoints) > 0:
            try:
                # Get the keypoints of the first detected person
                keypoints = results.keypoints[0].data[0].cpu().numpy()  # Shape: (17, 3)
                if keypoints.shape[0] > 0:
                    self.current_keypoints = keypoints
            except Exception as e:
                print(f"Error processing keypoints: {str(e)}")
                self.current_keypoints = None
            
            # Get bounding box
            boxes = results.boxes
            if len(boxes) > 0:
                box = boxes[0].xyxy[0].cpu().numpy()  # Get first person's box
                x1, y1, x2, y2 = map(int, box)
            
            # Draw keypoints and connections
            for kpt in keypoints:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(frame, (x, y), 5, (245,117,66), -1)
                
            # Draw skeleton connections
            skeleton = [
                (5,7), (7,9),   # Left arm
                (6,8), (8,10),  # Right arm
                (5,6),          # Shoulders
                (5,11), (6,12), # Torso
                (11,13), (13,15), # Left leg
                (12,14), (14,16)  # Right leg
            ]
            
            for connection in skeleton:
                pt1 = tuple(map(int, keypoints[connection[0]][:2]))
                pt2 = tuple(map(int, keypoints[connection[1]][:2]))
                cv2.line(frame, pt1, pt2, (245,66,230), 2)

            if target_keypoints is not None:
                similarity = self.calculate_pose_similarity(
                    self.current_keypoints,
                    target_keypoints
                )
                
                # Update best match if current similarity is higher
                if similarity > self.highest_similarity:
                    self.highest_similarity = similarity
                    self.best_frame = frame.copy()
                
                # Set colors and status based on similarity
                if similarity >= self.similarity_threshold:
                    draw_color = (0, 255, 0)  # Green
                    status = "PERFECT MATCH!"
                elif similarity >= self.similarity_threshold * 0.90:  # Close to matching
                    draw_color = (0, 255, 255)  # Yellow
                    status = "Almost there!"
                else:
                    draw_color = (0, 0, 255)  # Red
                    status = "Keep adjusting..."
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                
                # Suppress on-frame text overlays (similarity, best, status)
                # These details are displayed below the video in the UI.

        return frame, similarity, len(results.keypoints) > 0, self.best_frame

    def bbox_fall_or_stand(self, bbox):
        """Check if bounding box suggests fall."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return "STAND"
        return "FALL" if (w / h) > 1.6 else "STAND"

    def get_pose_landmarks(self, image_path):
        """Get pose keypoints from an image using YOLOv8."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                return None
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose_model(image_rgb, imgsz = 384)[0]
            
            if results.keypoints is not None and len(results.keypoints) > 0:
                # Get keypoints of the first person detected
                keypoints = results.keypoints[0].data[0].cpu().numpy()
                
                # Verify we have valid keypoints
                if keypoints.shape[0] > 0:
                    print(f"Found {keypoints.shape[0]} keypoints in {image_path}")
                    return keypoints
                else:
                    print(f"No valid keypoints found in {image_path}")
                    return None
            else:
                print(f"No pose detected in {image_path}")
                return None
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def get_torso_angle(self, landmarks, image_shape):
        """Calculate torso tilt angle."""
        ih, iw = image_shape[:2]
        ls, rs, lh, rh = landmarks[11], landmarks[12], landmarks[23], landmarks[24]
        
        x_ls, y_ls = int(ls.x * iw), int(ls.y * ih)
        x_rs, y_rs = int(rs.x * iw), int(rs.y * ih)
        x_lh, y_lh = int(lh.x * iw), int(lh.y * ih)
        x_rh, y_rh = int(rh.x * iw), int(rh.y * ih)
        
        mid_sh_x = (x_ls + x_rs) // 2
        mid_sh_y = (y_ls + y_rs) // 2
        mid_hip_x = (x_lh + x_rh) // 2
        mid_hip_y = (y_lh + y_rh) // 2
        
        vec_x = mid_sh_x - mid_hip_x
        vec_y = mid_sh_y - mid_hip_y
        mag_v = np.sqrt(vec_x ** 2 + vec_y ** 2)
        
        if mag_v < 1e-6:
            return 0.0, (mid_hip_x, mid_hip_y), (mid_sh_x, mid_sh_y)
            
        cos_theta = (-vec_y) / mag_v
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        return np.degrees(angle_rad), (mid_hip_x, mid_hip_y), (mid_sh_x, mid_sh_y)
