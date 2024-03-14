import cv2
import mediapipe as mp
import csv

def detect_pose_landmarks(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    
    # Process the image
    results = pose.process(image_rgb)
    
    # Prepare landmarks data
    landmarks_data = []
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks_data.append({
                "landmark_id": idx,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility,
            })
    
    # Save landmarks data to CSV file
    with open("landmarks.csv", mode="w", newline='') as file:
        fieldnames = ["landmark_id", "x", "y", "z", "visibility"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for data in landmarks_data:
            writer.writerow(data)
    
    # Draw landmarks on the image
    annotated_image = image.copy()
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    return annotated_image

def main():
    image_path = "samplePic.jpg"
    annotated_image = detect_pose_landmarks(image_path)
    
    while True:
        cv2.imshow("Pose Landmarks", annotated_image)
        key = cv2.waitKey(1)
        if key == 27: 
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
