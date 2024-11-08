import cv2
import torch
import numpy as np

class FaceBlur:
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize YOLOv5-face detection and blurring system
        """
        # Load YOLOv5 model directly from torch hub
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                  path='yolov5n.pt')
        self.model.conf = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
    def apply_gaussian_blur(self, image, box, kernel_size=99):
        """Apply Gaussian blur to detected face regions"""
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]
        
        # Ensure kernel size is odd
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 30)
        image[y1:y2, x1:x2] = blurred_roi
        return image

    def process_frame(self, frame):
        """Process a single frame to detect and blur faces"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.model(frame_rgb)
        
        # Get detections and apply blur
        if len(results.xyxy[0]) > 0:
            for detection in results.xyxy[0]:
                if detection[-2] > self.model.conf:
                    box = detection[:4].cpu().numpy()
                    frame = self.apply_gaussian_blur(frame, box)
        
        return frame

    def process_video(self, input_path, output_path):
        """Process video file to blur faces"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            processed_frame = self.process_frame(frame)
            out.write(processed_frame)

            # Display progress
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")

        # Release resources
        cap.release()
        out.release()
        print("Video processing completed!")

    def process_webcam(self):
        """Process webcam feed in real-time"""
        # Try different camera indices
        for camera_index in range(5):
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                break
        else:
            print("Unable to open camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display result
            cv2.imshow('Face Blur', processed_frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                    
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Initialize face blur system
    face_blur = FaceBlur(confidence_threshold=0.5)
    
    # Process webcam feed
    face_blur.process_webcam()

if __name__ == "__main__":
    main()