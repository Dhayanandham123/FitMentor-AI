#!/usr/bin/env python3
"""
AI Squat Trainer

- Uses MediaPipe Pose + OpenCV to track squat form.
- Counts correct / incorrect reps.
- Gives voice feedback via pyttsx3.
- Saves data into workout_data.json (daily + weekly, real ISO week).
- At the end of the session, calls:
    - DrillSuggestionAgent
    - ProgressAgent
    - SchedulingAgent
to generate drill advice, progress summary, and next-session plan.
"""

import cv2
import math
import time
import json
import pyttsx3
import mediapipe as mp
from datetime import datetime, date
import os

from agents import DrillSuggestionAgent, ProgressAgent, SchedulingAgent

# ==========================
#  PATHS
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "workout_data.json")

# ==========================
#  Helpers
# ==========================

def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) at point b
    given three points a, b, c (each is (x, y)).
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)

    if mag_ab == 0 or mag_cb == 0:
        return 180.0

    cos_angle = max(min(dot / (mag_ab * mag_cb), 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))


class VoiceCoach:
    def __init__(self, cooldown=3.0):
        self.engine = pyttsx3.init()
        self.last = {}
        self.cooldown = cooldown

    def say(self, key, text):
        now = time.time()
        if key in self.last and now - self.last[key] < self.cooldown:
            return
        self.last[key] = now
        self.engine.say(text)
        self.engine.runAndWait()


# ==========================
#  Squat classification logic
# ==========================

def classify_squat(landmarks, w, h):
    """
    Given mediapipe landmarks + frame size,
    return (is_correct, reasons_str, knee_angle, trunk_angle)
    evaluated at the bottom of the squat.
    """

    lm = landmarks

    def get_xy(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    hip      = get_xy(mp.solutions.pose.PoseLandmark.LEFT_HIP)
    knee     = get_xy(mp.solutions.pose.PoseLandmark.LEFT_KNEE)
    ankle    = get_xy(mp.solutions.pose.PoseLandmark.LEFT_ANKLE)
    shoulder = get_xy(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)
    foot     = get_xy(mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX)

    issues = []

    # 1) Knee angle ~90°
    knee_angle = calculate_angle(hip, knee, ankle)
    if not (80 <= knee_angle <= 110):
        issues.append(f"knee angle not around 90° (current ~{int(knee_angle)}°)")

    # 2) Trunk lean angle relative to vertical
    sx, sy = shoulder
    hx, hy = hip
    v1 = (sx - hx, sy - hy)
    v2 = (0, -1)  # vertical up

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag1 == 0:
        trunk_angle = 0
    else:
        cosang = max(min(dot / (mag1 * mag2), 1.0), -1.0)
        trunk_angle = math.degrees(math.acos(cosang))

    if trunk_angle < 5:
        issues.append("torso too upright – lean slightly forward")
    elif trunk_angle > 30:
        issues.append("torso too far forward – keep chest more upright")

    # 3) Knee vs toes (rough)
    kx, ky = knee
    fx, fy = foot
    if kx > fx + 40:
        issues.append("knee crosses too far over toes")

    is_correct = len(issues) == 0
    if is_correct:
        reasons = "Good squat: depth, torso and knees all look fine."
    else:
        reasons = "; ".join(issues)

    return is_correct, reasons, knee_angle, trunk_angle


# ==========================
#  Save workout data (REAL WEEK)
# ==========================

def save_workout_data(correct_reps, incorrect_reps):
    """
    Save squat data with real week tracking.
    Uses ISO week (1–53). Keeps last 4 weeks in the array.
    """
    today = date.today()
    iso_week = today.isocalendar()[1]

    try:
        try:
            with open(DATA_FILE, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {
                'daily': {
                    'squats':   [0]*7,
                    'pushups':  [0]*7
                },
                'weekly': {
                    'squats':   [0]*4,
                    'pushups':  [0]*4
                },
                'last_week': iso_week
            }

        data.setdefault('daily', {})
        data.setdefault('weekly', {})
        data['daily'].setdefault('squats',  [0]*7)
        data['daily'].setdefault('pushups', [0]*7)
        data['weekly'].setdefault('squats',  [0]*4)
        data['weekly'].setdefault('pushups', [0]*4)
        if 'last_week' not in data:
            data['last_week'] = iso_week

        # Week change
        if data['last_week'] != iso_week:
            data['weekly']['squats']  = data['weekly']['squats'][1:]  + [0]
            data['weekly']['pushups'] = data['weekly']['pushups'][1:] + [0]
            data['daily']['squats']   = [0]*7
            data['daily']['pushups']  = [0]*7
            data['last_week'] = iso_week

        # Update today's squats
        day_index = datetime.now().weekday()  # 0=Mon..6=Sun
        if len(data['daily']['squats']) < 7:
            data['daily']['squats'] = (data['daily']['squats'] + [0]*7)[:7]
        data['daily']['squats'][day_index] += correct_reps

        # Recompute weekly squats total
        data['weekly']['squats'][-1] = sum(data['daily']['squats'])

        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✅ Saved {correct_reps} correct squats for ISO week {iso_week} to {DATA_FILE}")

    except Exception as e:
        print("Error saving workout data:", e)


# ==========================
#  Main loop
# ==========================

def main():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    coach = VoiceCoach(cooldown=2.5)

    # Agents
    drill_agent = DrillSuggestionAgent("squats")
    progress_agent = ProgressAgent(DATA_FILE)
    schedule_agent = SchedulingAgent()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    correct_reps = 0
    incorrect_reps = 0

    state = "up"
    last_classification = None

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            panel_width = 400
            canvas = cv2.resize(frame, (w, h))
            full = cv2.copyMakeBorder(
                canvas, 0, 0, 0, panel_width,
                cv2.BORDER_CONSTANT,
                value=(40, 0, 80)
            )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            knee_angle_disp = None

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    full[:, :w, :],
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                )

                lm = results.pose_landmarks.landmark

                left_hip   = lm[mp_pose.PoseLandmark.LEFT_HIP]
                left_knee  = lm[mp_pose.PoseLandmark.LEFT_KNEE]
                left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]

                hip_xy   = (int(left_hip.x * w), int(left_hip.y * h))
                knee_xy  = (int(left_knee.x * w), int(left_knee.y * h))
                ankle_xy = (int(left_ankle.x * w), int(left_ankle.y * h))

                knee_angle = calculate_angle(hip_xy, knee_xy, ankle_xy)
                knee_angle_disp = knee_angle

                DOWN_TH = 110
                UP_TH   = 150

                if state == "up" and knee_angle < DOWN_TH:
                    is_correct, reasons, k_ang, t_ang = classify_squat(lm, w, h)
                    last_classification = (is_correct, reasons)
                    state = "down"

                elif state == "down" and knee_angle > UP_TH:
                    if last_classification is not None:
                        is_correct, reasons = last_classification
                        if is_correct:
                            correct_reps += 1
                            coach.say("good_rep", "Nice squat. Keep that form.")
                        else:
                            incorrect_reps += 1
                            coach.say("bad_rep", reasons)
                    last_classification = None
                    state = "up"

            x0 = w
            x1 = w + panel_width

            cv2.rectangle(full, (x0, 0), (x1, 60), (80, 255, 200), -1)
            cv2.putText(full, "Proper Squat", (x0 + 20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.rectangle(full, (x0 + 20, 80), (x0 + 180, 115), (0, 160, 0), -1)
            cv2.putText(full, f"Correct: {correct_reps}", (x0 + 25, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.rectangle(full, (x0 + 200, 80), (x0 + 380, 115), (0, 0, 200), -1)
            cv2.putText(full, f"Incorrect: {incorrect_reps}", (x0 + 205, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            y_text = 150
            cv2.putText(full, "CUES:", (x0 + 20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 255), 2)
            y_text += 30
            cv2.putText(full, "- Slightly lean forward,", (x0 + 20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            y_text += 25
            cv2.putText(full, "  chest up.", (x0 + 40, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            y_text += 25
            cv2.putText(full, "- Aim for ~90 deg at", (x0 + 20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            y_text += 25
            cv2.putText(full, "  the knees.", (x0 + 40, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            y_text += 25
            cv2.putText(full, "- Knees should not move", (x0 + 20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            y_text += 25
            cv2.putText(full, "  too far past toes.", (x0 + 40, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

            if knee_angle_disp is not None:
                cv2.putText(full, f"Knee angle: {int(knee_angle_disp)} deg",
                            (x0 + 20, h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 200, 0), 1)

            cv2.putText(full, "Press 'q' to quit and save", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("AI Squat Trainer", full)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n🎉 Workout Complete!")
    print(f"✅ Correct squats: {correct_reps}")
    print(f"❌ Incorrect squats: {incorrect_reps}")

    # -------- AGENTS --------
    # No numeric score for squats → avg_score = None
    session_summary = {
        "exercise": "squats",
        "correct": correct_reps,
        "incorrect": incorrect_reps,
        "avg_score": None,
        "timestamp": datetime.now().isoformat(),
    }

    drill = drill_agent.suggest(session_summary)

    # Save workout data
    save_workout_data(correct_reps, incorrect_reps)

    # Progress + scheduling
    progress = progress_agent.get_week_summary()
    next_plan = schedule_agent.next_session(recommended_focus=drill["focus"])

    print("\n📌 Drill Suggestion Agent:")
    print("   →", drill["message"])
    coach.say("drill", drill["message"])

    print("\n📈 Progress Agent (this week):")
    print(f"   Squats : {progress['this_week']['squats']} reps")
    print(f"   Pushups: {progress['this_week']['pushups']} reps")
    print(f"   Total  : {progress['this_week']['total']} reps")
    print("   Trend  :", progress['trend'])
    print("   Note   :", progress['note'])

    print("\n🗓 Scheduling Agent (next session):")
    print(f"   Day plan: {next_plan['day']} → {next_plan['plan']}")
    print("   Message :", next_plan['message'])

    print(f"\n🔄 Return to your browser to see updated charts!")


if __name__ == "__main__":
    main()
