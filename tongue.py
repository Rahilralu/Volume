import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Open webcam
cap = cv2.VideoCapture(0)

print("[INFO] Stick your tongue out to increase volume, close mouth to decrease.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Tongue detection: check distance between lips
            upper_lip_y = face_landmarks.landmark[13].y  # Upper lip center
            lower_lip_y = face_landmarks.landmark[14].y  # Lower lip center
            mouth_gap = lower_lip_y - upper_lip_y

            # Simple heuristic: if mouth gap > threshold → assume tongue out
            if mouth_gap > 0.05:
                # Increase volume
                current_vol = volume.GetMasterVolumeLevelScalar()
                volume.SetMasterVolumeLevelScalar(min(current_vol + 0.02, 1.0), None)
                cv2.putText(frame, "Tongue Out: Volume UP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Decrease volume
                current_vol = volume.GetMasterVolumeLevelScalar()
                volume.SetMasterVolumeLevelScalar(max(current_vol - 0.02, 0.0), None)
                cv2.putText(frame, "Mouth Closed: Volume DOWN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Tongue Volume Controller", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
