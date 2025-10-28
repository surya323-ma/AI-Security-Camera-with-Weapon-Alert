import cv2
import threading
import pygame
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = 'yolov8n.pt'  # You can replace with a custom-trained weapon model
TARGET_OBJECTS = ['knife', 'gun', 'pistol', 'rifle', 'weapon']
ALARM_SOUND_FILE = 'enemy.mp3'
# --- End Configuration ---

# Alarm flag
g_alarm_sounding = False

# Load YOLO model
model = YOLO(MODEL_PATH)

def play_alarm_sound():
    """
    Plays the alarm sound using pygame in a separate thread.
    """
    global g_alarm_sounding
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(ALARM_SOUND_FILE)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Error playing sound: {e}")
    g_alarm_sounding = False


# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("🔫 Weapon Detection Alarm System Running...")
print("🚨 Alarm will trigger if a weapon is detected.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame.")
        break

    # Run YOLO detection
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # Get detected class names
    detected_objects = [model.names[int(cls)] for cls in results[0].boxes.cls]

    # Check if any weapon-related object is detected
    target_detected = any(obj.lower() in TARGET_OBJECTS for obj in detected_objects)

    if target_detected and not g_alarm_sounding:
        print(f"🚨 ALERT! Weapon detected: {detected_objects}")
        g_alarm_sounding = True
        alarm_thread = threading.Thread(target=play_alarm_sound, daemon=True)
        alarm_thread.start()

    # Display the annotated video
    cv2.imshow("Weapon Detection Alarm", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ System stopped.")
