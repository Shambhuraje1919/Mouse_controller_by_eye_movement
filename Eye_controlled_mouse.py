import cv2
import mediapipe as mp
import pyautogui
import time

# ================== CONFIG ==================
CAMERA_INDEX = 0
BLINK_THRESHOLD = 0.008
CLICK_COOLDOWN = 0.6
SMOOTHING_TIME = 0.05
EXIT_KEY = 27  # ESC

IRIS_LANDMARKS = range(474, 478)
RIGHT_EYE_LANDMARKS = (145, 159)

# ============================================


def init_camera(index):
    cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        raise RuntimeError("Camera not accessible")
    return cam


def init_facemesh():
    return mp.solutions.face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def move_cursor(landmark, frame_w, frame_h, screen_w, screen_h):
    screen_x = int(screen_w * landmark.x)
    screen_y = int(screen_h * landmark.y)
    pyautogui.moveTo(screen_x, screen_y, duration=SMOOTHING_TIME)


def draw_landmark(frame, landmark, frame_w, frame_h, color, radius=2):
    x = int(landmark.x * frame_w)
    y = int(landmark.y * frame_h)
    cv2.circle(frame, (x, y), radius, color, -1)


def is_blinking(landmarks):
    upper, lower = landmarks[RIGHT_EYE_LANDMARKS[0]], landmarks[RIGHT_EYE_LANDMARKS[1]]
    return abs(upper.y - lower.y) < BLINK_THRESHOLD


def main():
    cam = init_camera(CAMERA_INDEX)
    face_mesh = init_facemesh()
    screen_w, screen_h = pyautogui.size()

    last_click_time = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)

        frame_h, frame_w, _ = frame.shape

        if output.multi_face_landmarks:
            landmarks = output.multi_face_landmarks[0].landmark

            # Cursor movement using iris
            for idx, lm_index in enumerate(IRIS_LANDMARKS):
                landmark = landmarks[lm_index]
                draw_landmark(frame, landmark, frame_w, frame_h, (250, 250, 160))
                if idx == 1:
                    move_cursor(landmark, frame_w, frame_h, screen_w, screen_h)

            # Blink detection
            for lm in RIGHT_EYE_LANDMARKS:
                draw_landmark(frame, landmarks[lm], frame_w, frame_h, (0, 0, 255), 4)

            if is_blinking(landmarks):
                current_time = time.time()
                if current_time - last_click_time > CLICK_COOLDOWN:
                    pyautogui.click()
                    last_click_time = current_time

        cv2.imshow("Eye Controlled Mouse", frame)

        if cv2.waitKey(1) & 0xFF == EXIT_KEY:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
