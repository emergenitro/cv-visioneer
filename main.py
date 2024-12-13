import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize variables
prev_action = None
action_time = time.time()  # Time when the last action was taken
action_delay = 0.5  # Delay between actions in seconds

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Flip the frame to prevent mirroring
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    # Convert the image to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the RGB image to find hands
    results = hands.process(img_rgb)

    finger_counts = {}  # To store finger counts for each hand

    if results.multi_hand_landmarks:
        for hand_landmark, hand_info in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            # Get hand label (Left or Right)
            hand_label = hand_info.classification[0].label
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
            # Get hand landmarks
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)
            # Counting fingers
            finger_tips = [
                4,
                8,
                12,
                16,
                20,
            ]  # Thumb tip, Index tip, Middle tip, Ring tip, Pinky tip
            finger_status = []

            # Check if fingers are up
            # For Thumb
            if hand_label == "Right":
                if lm_list[finger_tips[0]].x < lm_list[finger_tips[0] - 1].x:
                    finger_status.append(1)
                else:
                    finger_status.append(0)
            else:
                if lm_list[finger_tips[0]].x > lm_list[finger_tips[0] - 1].x:
                    finger_status.append(1)
                else:
                    finger_status.append(0)
            # For other fingers
            for tip_id in finger_tips[1:]:
                if lm_list[tip_id].y < lm_list[tip_id - 2].y:
                    finger_status.append(1)
                else:
                    finger_status.append(0)
            total_fingers = finger_status.count(1)
            finger_counts[hand_label] = total_fingers

            # Display number of fingers
            cv2.putText(
                frame,
                f"{hand_label} Hand: {total_fingers}",
                (10, 50 if hand_label == "Left" else 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

    # Decide action based on finger counts
    action = "No action"
    current_time = time.time()
    if ("Left" in finger_counts) and ("Right" in finger_counts):
        left_count = finger_counts["Left"]
        right_count = finger_counts["Right"]
        if left_count == 1 and right_count == 1:
            action = "Jump"
        elif left_count == 2 and right_count == 2:
            action = "Slide"
    elif "Left" in finger_counts:
        left_count = finger_counts["Left"]
        if left_count == 1:
            action = "Move Left"
    elif "Right" in finger_counts:
        right_count = finger_counts["Right"]
        if right_count == 1:
            action = "Move Right"

    # Send the action if it has changed and cooldown has passed
    if action != prev_action and (current_time - action_time) > action_delay:
        if action == "Move Left":
            pyautogui.press("left")
            print("Move Left")
        elif action == "Move Right":
            pyautogui.press("right")
            print("Move Right")
        elif action == "Jump":
            pyautogui.press("up")
        elif action == "Slide":
            pyautogui.press("down")
        action_time = current_time
        prev_action = action
    elif action == "No action" and (current_time - action_time) > action_delay:
        prev_action = action

    # Display the action
    cv2.putText(
        frame,
        f"Action: {action}",
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Show the frame
    cv2.imshow("Hand Gesture Control", frame)

    # Make image on top of all windows
    cv2.setWindowProperty("Hand Gesture Control", cv2.WND_PROP_TOPMOST, 1)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release and destroy all windows
cap.release()
cv2.destroyAllWindows()
