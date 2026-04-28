import http.server
import socketserver
import json
import os

PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class APIHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/force_retrain':
            # Create a signal flag for the Live Pipeline daemon to execute the retrain safely
            flag_path = os.path.join(BASE_DIR, "outputs", "retrain.flag")
            with open(flag_path, "w") as f:
                f.write("trigger")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "success", "message": "Signal sent"}')
            
        elif self.path == '/api/predict_batch':
            # Read JSON payload and pipe straight to python inference engine
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            import subprocess
            import sys
            script_path = os.path.join(BASE_DIR, "src", "predict_live.py")
            
            # Use stdin to pass huge payloads securely
            process = subprocess.Popen([sys.executable, script_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(input=post_data)
            
            if stderr:
                # Log the crash to the terminal so the user/auditor can see it
                print(f"\n[AI ENGINE ERROR]\n{stderr}\n")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(stdout.encode('utf-8'))
            
        else:
            self.send_error(404, "Endpoint not found")

class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True

if __name__ == "__main__":
    os.chdir(BASE_DIR)
    with ThreadedHTTPServer(("", PORT), APIHandler) as httpd:
        httpd.serve_forever()
