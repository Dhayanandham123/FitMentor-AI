#!/usr/bin/env python3
"""
Flask backend server for fitness tracker.
Handles workout sessions, chatbot responses, and serves the dashboard.
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

import json
import os
import subprocess
import threading
import signal
import sys

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "workout_data.json")

# -------------------------------
# GEMINI CHATBOT
# -------------------------------
try:
    from google.genai import Client
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# SIMPLE VERSION: hard-code your API key HERE (backend only)
# >>> REPLACE "YOUR_GEMINI_API_KEY_HERE" with your real key string <<<
GEMINI_API_KEY = "AIzaSyA_3Mznxe6uA7YLlmLUKd0JP6Cv_XiWqDU".strip()

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    gemini_client = Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None

app = Flask(__name__, static_folder='.')
CORS(app)

active_process = None
active_lock = threading.Lock()

# =====================================================
# FRONTEND ROUTES
# =====================================================

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(BASE_DIR, path)

# =====================================================
# WORKOUT DATA
# =====================================================

@app.route('/api/workout_data', methods=['GET'])
def get_workout_data():
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except FileNotFoundError:
        default_data = {
            'daily': {
                'squats': [0] * 7,
                'pushups': [0] * 7
            },
            'weekly': {
                'squats': [0] * 4,
                'pushups': [0] * 4
            },
            'currentDay': 0,
            'currentWeek': 3
        }
        return jsonify(default_data)

# =====================================================
# START WORKOUT
# =====================================================

@app.route('/api/start_workout', methods=['POST'])
def start_workout():
    global active_process
    
    data = request.json or {}
    exercise_type = data.get('type')

    if exercise_type not in ['squats', 'pushups']:
        return jsonify({'error': 'Invalid exercise type'}), 400

    with active_lock:
        if active_process and active_process.poll() is None:
            return jsonify({'error': 'A workout is already in progress'}), 409

        script = 'squat.py' if exercise_type == 'squats' else 'pushup.py'
        script_path = os.path.join(BASE_DIR, script)

        if not os.path.exists(script_path):
            return jsonify({'error': f'Script not found: {script}'}), 500

        try:
            active_process = subprocess.Popen(
                [sys.executable, script_path],
                cwd=BASE_DIR
            )

            return jsonify({
                'success': True,
                'message': f'{exercise_type.capitalize()} workout started!',
                'pid': active_process.pid
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

# =====================================================
# WORKOUT STATUS
# =====================================================

@app.route('/api/workout_status', methods=['GET'])
def workout_status():
    global active_process

    with active_lock:
        if active_process is None:
            return jsonify({'running': False})

        if active_process.poll() is None:
            return jsonify({'running': True, 'pid': active_process.pid})
        else:
            active_process = None
            return jsonify({'running': False})

# =====================================================
# STOP WORKOUT
# =====================================================

@app.route('/api/stop_workout', methods=['POST'])
def stop_workout():
    global active_process

    with active_lock:
        if active_process and active_process.poll() is None:
            try:
                active_process.terminate()
                active_process.wait(timeout=5)
                active_process = None
                return jsonify({'success': True, 'message': 'Workout stopped'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        return jsonify({'error': 'No active workout'}), 404

# =====================================================
# GEMINI CHATBOT API
# =====================================================

@app.route("/api/chat", methods=["POST"])
def chat():
    if not GEMINI_AVAILABLE:
        return jsonify({"reply": "❌ Server missing google-genai. Run: pip install google-genai"}), 500

    if not GEMINI_API_KEY:
        return jsonify({"reply": "⚠️ No Gemini API key configured in app.py"}), 500

    if gemini_client is None:
        return jsonify({"reply": "❌ Gemini client not initialized."}), 500

    try:
        user_msg = (request.json or {}).get("message", "").strip()
        if not user_msg:
            return jsonify({"reply": "Please type a message first."}), 400

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "You are a professional fitness coach AI. "
                                "Answer in a friendly, concise way and focus on workouts, form, "
                                "reps, safety, and basic nutrition. User message: " + user_msg
                            )
                        }
                    ]
                }
            ]
        )

        text = getattr(response, "text", None) or "No response from Gemini."
        return jsonify({"reply": text})

    except Exception as e:
        return jsonify({"reply": f"❌ Error: {str(e)}"}), 500

# =====================================================
# INIT DATA
# =====================================================

def init_workout_data():
    if not os.path.exists(DATA_FILE):
        default_data = {
            'daily': {'squats': [0] * 7, 'pushups': [0] * 7},
            'weekly': {'squats': [0] * 4, 'pushups': [0] * 4},
            'currentDay': 0,
            'currentWeek': 3
        }
        with open(DATA_FILE, 'w') as f:
            json.dump(default_data, f, indent=2)

# =====================================================
# SHUTDOWN
# =====================================================

def signal_handler(sig, frame):
    global active_process
    print('\n🛑 Stopping server...')

    with active_lock:
        if active_process and active_process.poll() is None:
            active_process.terminate()
            active_process.wait(timeout=5)

    print('✅ Shutdown complete.')
    sys.exit(0)

# =====================================================

if __name__ == '__main__':
    init_workout_data()
    signal.signal(signal.SIGINT, signal_handler)

    print("\n🏋️ FITNESS TRACKER RUNNING → http://localhost:5000\n")

    app.run(debug=False, host='0.0.0.0', port=5000)
