import os
import subprocess
import sys
import time
import webbrowser

PORT = 8000

if __name__ == "__main__":
    # Ensure we are in the script's directory so paths resolve correctly
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)
    
    print("==================================================================================")
    print("                    WEATHER ML PROJECT: DASHBOARD & PIPELINE                      ")
    print("==================================================================================")
    
    print("\n[SYSTEM] Starting Local Web Server in background...")
    
    # Start the server as a background process
    # We use 'python -m http.server' to ensure CORS works when accessed via localhost
    try:
        server_process = subprocess.Popen(
            [sys.executable, "server.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(2) # Give it a moment to bind
        
        # Check if process is still running (it might die immediately if port is in use)
        if server_process.poll() is not None:
            raise RuntimeError(f"Port {PORT} is likely already in use by another process.")
            
        print(f"[OK] Server running at: http://localhost:{PORT}/dashboard/index.html")
        # WE WILL OPEN THE BROWSER AFTER THE PIPELINE FINISHES
    except Exception as e:
        print(f"[ERROR] Could not start server: {e}")
        print("[TIP] Try closing other Python windows or run 'taskkill /F /IM python.exe' if stuck.")
        sys.exit(1)

    print("\n[STEP 1] EXECUTING MACHINE LEARNING PIPELINE\n")
    
    # Ask user about retraining before launching the subprocess
    retrain_base = input("   Do you want to retrain base models? (y/n): ").strip().lower()
    retune = input("   Do you want to re-tune hyperparameters? (y/n): ").strip().lower()
    
    # Pass choices to the subprocess via environment variables
    env = os.environ.copy()
    env['RETRAIN_BASE_MODELS'] = retrain_base
    env['RETUNE_MODELS'] = retune
    
    try:
        # Run the ML Pipeline with LIVE output
        process = subprocess.Popen(
            [sys.executable, "src/weather_ml.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end="")
            
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)
            
        print("\n[OK] Pipeline completed successfully.")
        
        # NOW open the browser
        print("\n[SYSTEM] Opening Dashboard...")
        webbrowser.open_new_tab(f"http://localhost:{PORT}/dashboard/index.html")
        
        # [STEP 2] STARTING LIVE PIPELINE & CATCH-UP ENGINE
        print("\n[STEP 2] STARTING LIVE PIPELINE & CATCH-UP ENGINE\n")
        live_process = subprocess.Popen(
            [sys.executable, "src/live_pipeline.py"],
            stdout=sys.stdout,   
            stderr=sys.stderr
        )
        print("[OK] Background Catch-Up Engine started.")
        
    except subprocess.CalledProcessError:
        print("\n[X] The ML Pipeline failed. Review the logs above.")
        server_process.terminate()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user.")
        server_process.terminate()
        if 'live_process' in locals():
            live_process.terminate()
        sys.exit(1)

    print("\n[STEP 3] DASHBOARD IS LIVE")
    print(f"To interact with results, visit: http://localhost:{PORT}/dashboard/index.html")
    print("KEEP THIS TERMINAL OPEN to maintain the connection and watch the Catch-Up engine retrain.")
    print("\n(Press Ctrl+C to stop both the dashboard server and background engine)\n")
    
    try:
        # Avoid blocking .wait() which can hide crashes
        while True:
            time.sleep(1)
            if server_process.poll() is not None:
                print("\n[!] Web Server process died unexpectedly.")
                break
            if 'live_process' in locals() and live_process.poll() is not None:
                print("\n[!] Live Pipeline process died unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down server and live engine...")
        server_process.terminate()
        if 'live_process' in locals():
            live_process.terminate()
        print("[DONE] Project closed.")
