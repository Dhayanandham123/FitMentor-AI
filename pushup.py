#!/usr/bin/env python3
"""
AI Push-up Trainer

- Uses MediaPipe Pose + OpenCV to track push-up plank quality.
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
#  Math helpers
# ==========================

def angle(a, b, c):
    """
    Returns angle in degrees at point b formed by points a-b-c.
    a, b, c are (x, y).
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

    cosang = max(min(dot / (mag_ab * mag_cb), 1.0), -1.0)
    return math.degrees(math.acos(cosang))


# ==========================
#  Voice coach
# ==========================

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
#  Push-up classification
# ==========================

def classify_pushup(landmarks, w, h):
    """
    Use one body side to estimate push-up plank quality.

    Returns:
      score (0–100), is_good (bool), msg (str),
      body_angle (float), leg_angle (float),
      key_points (shoulder, hip, knee, ankle)
    """
    lm = landmarks

    def xy(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    shoulder = xy(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)
    hip      = xy(mp.solutions.pose.PoseLandmark.RIGHT_HIP)
    knee     = xy(mp.solutions.pose.PoseLandmark.RIGHT_KNEE)
    ankle    = xy(mp.solutions.pose.PoseLandmark.RIGHT_ANKLE)

    body_angle = angle(shoulder, hip, ankle)  # 180 = straight
    leg_angle  = angle(hip, knee, ankle)

    # Map angles into [0,1] scores
    score_body = max(0.0, min(1.0, (body_angle - 150) / 30.0))
    score_leg  = max(0.0, min(1.0, (leg_angle - 150) / 25.0))

    score = (0.6 * score_body + 0.4 * score_leg) * 100.0
    score = max(0.0, min(100.0, score))
    is_good = score >= 80

    if not is_good:
        reasons = []
        if body_angle < 170:
            reasons.append("keep hips in line with shoulders and ankles")
        if leg_angle < 165:
            reasons.append("straighten your knees and legs")
        msg = ", ".join(reasons) if reasons else "Keep your body in one long line."
    else:
        msg = "Nice one, long line body. Keep going."

    return score, is_good, msg, body_angle, leg_angle, (shoulder, hip, knee, ankle)


# ==========================
#  Save workout data (REAL WEEK)
# ==========================

def save_workout_data(correct_reps, incorrect_reps):
    """
    Save push-up data with real week tracking.
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

        # Ensure structure
        data.setdefault('daily', {})
        data.setdefault('weekly', {})
        data['daily'].setdefault('squats',  [0]*7)
        data['daily'].setdefault('pushups', [0]*7)
        data['weekly'].setdefault('squats',  [0]*4)
        data['weekly'].setdefault('pushups', [0]*4)
        if 'last_week' not in data:
            data['last_week'] = iso_week

        # Week change detection
        if data['last_week'] != iso_week:
            # shift 4-week history: [w2,w3,w4,0]
            data['weekly']['squats']  = data['weekly']['squats'][1:]  + [0]
            data['weekly']['pushups'] = data['weekly']['pushups'][1:] + [0]
            # reset daily for new week
            data['daily']['squats']   = [0]*7
            data['daily']['pushups']  = [0]*7
            data['last_week'] = iso_week

        # Update today's pushups
        day_index = datetime.now().weekday()  # 0=Mon ... 6=Sun
        if len(data['daily']['pushups']) < 7:
            data['daily']['pushups'] = (data['daily']['pushups'] + [0]*7)[:7]
        data['daily']['pushups'][day_index] += correct_reps

        # Recompute weekly total (only pushups here)
        data['weekly']['pushups'][-1] = sum(data['daily']['pushups'])

        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✅ Saved {correct_reps} correct push-ups for ISO week {iso_week} to {DATA_FILE}")

    except Exception as e:
        print(f"Error saving workout data: {e}")


# ==========================
#  Main loop
# ==========================

def main():
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    coach = VoiceCoach(cooldown=3.0)

    # Agents
    drill_agent = DrillSuggestionAgent("pushups")
    progress_agent = ProgressAgent(DATA_FILE)
    schedule_agent = SchedulingAgent()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    coach.say(
        "start",
        "Starting push up session. Keep your hands under your shoulders and your body in a straight line."
    )

    rep_state = "top"
    correct_reps = 0
    incorrect_reps = 0

    # for a rough average score
    score_sum = 0.0
    score_count = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            score = None
            body_angle_val = None
            leg_angle_val = None
            key_points = None
            is_good = False
            msg = ""

            if res.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                )

                score, is_good, msg, body_angle_val, leg_angle_val, key_points = classify_pushup(
                    res.pose_landmarks.landmark, w, h
                )

                if score is not None:
                    score_sum += score
                    score_count += 1

                lm = res.pose_landmarks.landmark
                r_sh = int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)
                r_el = int(lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h)

                # simple rep state machine based on elbow height
                if rep_state == "top" and r_el > r_sh + 40:
                    rep_state = "bottom"
                elif rep_state == "bottom" and r_el < r_sh + 15:
                    rep_state = "top"
                    if is_good:
                        correct_reps += 1
                        coach.say("good_rep", "Nice push up. Keep that strong plank line.")
                    else:
                        incorrect_reps += 1
                        coach.say("bad_rep", msg)

            overlay = frame.copy()

            if key_points is not None:
                shoulder, hip, knee, ankle = key_points
                pts = [ankle, knee, hip, shoulder]
                for p in pts:
                    cv2.circle(overlay, p, 8, (255, 0, 255), -1)
                for i in range(len(pts) - 1):
                    cv2.line(overlay, pts[i], pts[i+1], (255, 255, 255), 3)

                mid_body = ((hip[0] + knee[0]) // 2, (hip[1] + knee[1]) // 2)
                cv2.putText(
                    overlay,
                    f"{int(body_angle_val) if body_angle_val else 0} deg",
                    (mid_body[0] - 40, mid_body[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            banner_h = 60
            if is_good:
                color = (0, 200, 0)
                text = "Nice one – Long line body"
            else:
                color = (0, 0, 255)
                text = "Attention! Not perfect long line body"

            cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
            cv2.putText(
                overlay,
                text,
                (20, int(banner_h * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.rectangle(overlay, (0, banner_h), (200, banner_h + 140), (0, 0, 255), -1)
            cv2.putText(
                overlay,
                "Push up",
                (10, banner_h + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"Correct: {correct_reps}",
                (10, banner_h + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"Incorrect: {incorrect_reps}",
                (10, banner_h + 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 100, 100),
                2,
                cv2.LINE_AA,
            )

            if score is not None:
                bar_w = 60
                x0 = w - bar_w - 20
                y0 = banner_h + 20
                y1 = h - 40

                cv2.rectangle(overlay, (x0, y0), (x0 + bar_w, y1), (0, 100, 0), 2)

                pct = int(score)
                total_h = y1 - y0 - 4
                filled_h = int(total_h * pct / 100.0)
                cv2.rectangle(
                    overlay,
                    (x0 + 2, y1 - filled_h),
                    (x0 + bar_w - 2, y1 - 2),
                    (0, 255, 0),
                    -1,
                )

                cv2.putText(
                    overlay,
                    f"{pct} %",
                    (x0 - 10, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            base_y = h - 80
            cv2.putText(
                overlay,
                "CUES:",
                (20, base_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                overlay,
                "- Hands under shoulders",
                (20, base_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                1,
            )
            cv2.putText(
                overlay,
                "- Body in one long line",
                (20, base_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                1,
            )
            cv2.putText(
                overlay,
                "- Lower chest close to floor",
                (20, base_y + 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                1,
            )

            cv2.putText(
                overlay,
                "Press 'q' to quit and save",
                (w - 280, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("AI Push-up Trainer", overlay)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n🎉 Workout Complete!")
    print(f"✅ Correct push-ups: {correct_reps}")
    print(f"❌ Incorrect push-ups: {incorrect_reps}")

    # -------- AGENTS --------
    avg_score = (score_sum / score_count) if score_count > 0 else None

    session_summary = {
        "exercise": "pushups",
        "correct": correct_reps,
        "incorrect": incorrect_reps,
        "avg_score": avg_score,
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
