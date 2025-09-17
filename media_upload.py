import cv2
from deepface import DeepFace
import tkinter as tk
from tkinter import filedialog

def run_media_upload():
    # Initialize file dialog
    root = tk.Tk()
    root.update()
    root.withdraw()
    root.attributes('-topmost', True)

    # Ask user to select a media file
    media_path = filedialog.askopenfilename(
        title="Select an image or video file",
        filetypes=[("Media files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov *.mkv")]
    )

    if not media_path:
        print("❌ No file selected.")
        return

    # Emotion detection function
    def detect_emotions(frame):
        try:
            # Resize frame for consistent detection
            resized_frame = cv2.resize(frame, (640, 480))
            results = DeepFace.analyze(resized_frame, actions=['emotion'], enforce_detection=False)

            # Ensure results is a list
            if isinstance(results, dict):
                results = [results]

            # Draw each face and emotion on resized frame
            for result in results:
                x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
                emotion = result['dominant_emotion']
                cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(resized_frame, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            return resized_frame

        except Exception as e:
            print("Detection error:", e)
            return frame

    # Handle image files
    if media_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        frame = cv2.imread(media_path)
        if frame is not None:
            output = detect_emotions(frame)
            cv2.imshow("Emotion Detection - Image", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("❌ Failed to load image.")

    # Handle video files
    elif media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(media_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            output = detect_emotions(frame)
            cv2.imshow("Emotion Detection - Video", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# Prevent auto-run when imported
if __name__ == "__main__":
    run_media_upload()
