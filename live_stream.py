import cv2
from deepface import DeepFace

def run_live_stream():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Webcam not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Resize frame for better performance
            resized_frame = cv2.resize(frame, (640, 480))

            # Analyze emotions
            results = DeepFace.analyze(resized_frame, actions=['emotion'], enforce_detection=False)

            # Ensure results is a list
            if isinstance(results, dict):
                results = [results]

            # Show face count
            cv2.putText(frame, f"Faces detected: {len(results)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Loop through each detected face
            for result in results:
                x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
                emotion = result['dominant_emotion']

                # Adjust coordinates if needed
                x = max(x, 0)
                y = max(y, 0)

                # Draw bounding box and emotion label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        except Exception as e:
            print("Detection error:", e)
            cv2.putText(frame, "Detection error", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Live Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_stream()
