import subprocess
import time
import os
import sys

def main():
    port = os.getenv("PORT", "8080")
    print(f"Starting unified Fraud Intelligence Platform on port {port}...")
    
    # 1. Start FastAPI backend server on localhost:8000
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"]
    )
    
    # Give backend a moment to bind
    time.sleep(2)
    
    # 2. Start Streamlit frontend server on 0.0.0.0:${PORT}
    frontend_proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", port,
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ],
        env=dict(os.environ, API_URL="http://127.0.0.1:8000")
    )
    
    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        backend_proc.terminate()
        frontend_proc.terminate()

if __name__ == "__main__":
    main()
