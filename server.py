#!/usr/bin/env python3
"""
Simple HTTP server to serve the fitness tracker dashboard.
This allows the HTML to read the workout_data.json file.
"""
import http.server
import socketserver
import json
import os
from urllib.parse import urlparse, parse_qs

PORT = 8000

class FitnessServerHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow JSON loading
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

    def do_GET(self):
        # Parse the URL
        parsed_path = urlparse(self.path)
        
        # Serve workout data API endpoint
        if parsed_path.path == '/api/workout_data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            try:
                with open('workout_data.json', 'r') as f:
                    data = json.load(f)
                self.wfile.write(json.dumps(data).encode())
            except FileNotFoundError:
                # Return empty data structure if file doesn't exist yet
                default_data = {
                    'daily': {
                        'squats': [0, 0, 0, 0, 0, 0, 0],
                        'pushups': [0, 0, 0, 0, 0, 0, 0]
                    },
                    'weekly': {
                        'squats': [0, 0, 0, 0],
                        'pushups': [0, 0, 0, 0]
                    },
                    'currentDay': 0,
                    'currentWeek': 3
                }
                self.wfile.write(json.dumps(default_data).encode())
        else:
            # Serve regular files (HTML, CSS, JS)
            super().do_GET()

def main():
    # Create workout_data.json if it doesn't exist
    if not os.path.exists('workout_data.json'):
        default_data = {
            'daily': {
                'squats': [0, 0, 0, 0, 0, 0, 0],
                'pushups': [0, 0, 0, 0, 0, 0, 0]
            },
            'weekly': {
                'squats': [0, 0, 0, 0],
                'pushups': [0, 0, 0, 0]
            },
            'currentDay': 0,
            'currentWeek': 3
        }
        with open('workout_data.json', 'w') as f:
            json.dump(default_data, f, indent=2)
    
    with socketserver.TCPServer(("", PORT), FitnessServerHandler) as httpd:
        print("=" * 60)
        print("🏋️  FITNESS TRACKER SERVER STARTED")
        print("=" * 60)
        print(f"\n📊 Dashboard: http://localhost:{PORT}/index.html")
        print(f"\n📝 Instructions:")
        print(f"   1. Open http://localhost:{PORT}/index.html in your browser")
        print(f"   2. Click Squats or Push-ups card")
        print(f"   3. Run the command shown (python squat.py or python pushup.py)")
        print(f"   4. Complete your workout and press 'q' to save")
        print(f"   5. Return to browser to see updated charts!")
        print(f"\n🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        print()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n✅ Server stopped. Good workout!")

if __name__ == "__main__":
    main()