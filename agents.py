#!/usr/bin/env python3
"""
agents.py

Contains three simple "agents" that operate on your workout data:

1. DrillSuggestionAgent  – suggests the next drill after a session
2. ProgressAgent         – summarizes your current progress
3. SchedulingAgent       – suggests what to do on the next day
"""

import json
import os
from datetime import datetime, date


class DrillSuggestionAgent:
    """
    Uses last session performance (correct vs incorrect reps, quality)
    to suggest what you should focus on next.
    """

    def __init__(self, exercise: str):
        # "squats" or "pushups"
        self.exercise = exercise

    def suggest(self, session_summary: dict) -> dict:
        """
        session_summary:
            {
                "exercise": "pushups" | "squats",
                "correct": int,
                "incorrect": int,
                "avg_score": float | None,
                "timestamp": "...",
            }
        """
        correct = session_summary.get("correct", 0)
        incorrect = session_summary.get("incorrect", 0)
        avg_score = session_summary.get("avg_score", None)

        total = correct + incorrect
        if total == 0:
            quality = 0.0
        else:
            quality = correct / float(total)

        # Default suggestion
        focus = "balanced"
        text = "Nice work. Keep practicing with similar sets."

        # Very rough rule logic
        if total < 10:
            focus = "volume"
            text = (
                "You did very few reps. Next session, aim for more total reps "
                "with easier tempo and focus on consistency."
            )
        elif quality < 0.6:
            focus = "form"
            text = (
                "Many reps were marked incorrect. Next drill: reduce speed, "
                "shorten set size, and focus on cleaner technique."
            )
        elif quality >= 0.9:
            focus = "intensity"
            text = (
                "Your form looks solid. Next drill: increase difficulty – "
                "either more reps per set or slower tempo."
            )
        else:
            focus = "balanced"
            text = (
                "Good mix of correct reps. Next drill: keep similar rep range, "
                "but try to slightly improve your form and breathing."
            )

        if avg_score is not None:
            text += f" Your average quality score was about {avg_score:.0f}%."

        return {
            "focus": focus,     # "form" / "volume" / "intensity" / "balanced"
            "message": text,
        }


class ProgressAgent:
    """
    Reads workout_data.json and computes simple progress stats.
    """

    def __init__(self, data_file: str):
        self.data_file = data_file

    def _load(self):
        if not os.path.exists(self.data_file):
            return None
        with open(self.data_file, "r") as f:
            return json.load(f)

    def get_week_summary(self) -> dict:
        """
        Returns a simple progress summary like:
        {
            "this_week": {
                "squats": 42,
                "pushups": 35,
                "total": 77,
            },
            "trend": "up" | "down" | "flat",
            "note": "..."
        }
        """
        data = self._load()
        if not data:
            return {
                "this_week": {"squats": 0, "pushups": 0, "total": 0},
                "trend": "unknown",
                "note": "No data yet. Do your first session!",
            }

        daily = data.get("daily", {})
        weekly = data.get("weekly", {})

        squats_daily = daily.get("squats", [0] * 7)
        pushups_daily = daily.get("pushups", [0] * 7)

        squats_this_week = sum(squats_daily)
        pushups_this_week = sum(pushups_daily)
        total_this_week = squats_this_week + pushups_this_week

        squats_week_arr = weekly.get("squats", [0] * 4)
        pushups_week_arr = weekly.get("pushups", [0] * 4)

        # Last week vs previous week (rough)
        if len(squats_week_arr) >= 2 and len(pushups_week_arr) >= 2:
            prev_total = squats_week_arr[-2] + pushups_week_arr[-2]
            curr_total = squats_week_arr[-1] + pushups_week_arr[-1]
        else:
            prev_total = 0
            curr_total = total_this_week

        if prev_total == 0 and curr_total == 0:
            trend = "flat"
            note = "No activity recorded in the last two weeks."
        elif prev_total == 0 and curr_total > 0:
            trend = "up"
            note = "Nice, this is your first active week!"
        else:
            delta = curr_total - prev_total
            if delta > 10:
                trend = "up"
                note = "You increased your total reps compared to last week."
            elif delta < -10:
                trend = "down"
                note = "You did fewer reps than last week. Try to be more consistent."
            else:
                trend = "flat"
                note = "Your total volume is similar to last week."

        return {
            "this_week": {
                "squats": squats_this_week,
                "pushups": pushups_this_week,
                "total": total_this_week,
            },
            "trend": trend,
            "note": note,
        }


class SchedulingAgent:
    """
    Very simple day-based scheduler.

    Uses weekday (0=Mon..6=Sun) + trend/focus to decide:
      - what you should train next
      - whether to suggest rest
    """

    def __init__(self):
        self.plan = {
            0: "squats",              # Monday
            1: "pushups",             # Tuesday
            2: "both",                # Wednesday
            3: "squats",              # Thursday
            4: "pushups",             # Friday
            5: "light",               # Saturday
            6: "rest",                # Sunday
        }

    def next_session(self, recommended_focus: str = "balanced") -> dict:
        """
        Decide next session based on weekday + focus.

        Returns:
            {
                "day": "Monday",
                "plan": "squats" | "pushups" | "both" | "rest" | "light",
                "message": "..."
            }
        """
        today = date.today()
        weekday = today.weekday()   # 0..6
        names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        plan_today = self.plan.get(weekday, "both")

        # Build message
        if plan_today == "rest":
            msg = "Today is a programmed rest day. Focus on stretching and sleep."
        elif plan_today == "light":
            msg = "Today do a short, light session to keep the habit without overloading."
        elif plan_today == "both":
            msg = "Today is a full session: do both squats and push-ups."
        else:
            msg = f"Today focus mainly on {plan_today}."

        # Modify message slightly based on focus
        if recommended_focus == "form":
            msg += " Use easier sets and really focus on clean form."
        elif recommended_focus == "volume":
            msg += " Try to increase total reps with good pacing."
        elif recommended_focus == "intensity":
            msg += " You can safely push a little harder today."

        return {
            "day": names[weekday],
            "plan": plan_today,
            "message": msg,
        }
