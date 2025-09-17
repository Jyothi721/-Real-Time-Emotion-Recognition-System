from live_stream import run_live_stream
from media_upload import run_media_upload

def main():
    print("\n🎥 Emotion Recognition System")
    print("1. Live Webcam Detection")
    print("2. Media File Detection")

    choice = input("Select option (1 or 2): ")

    if choice == '1':
        print("\nStarting live webcam detection...")
        run_live_stream()

    elif choice == '2':
        print("\nOpening file explorer for media upload...")
        run_media_upload()

    else:
        print("❌ Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
