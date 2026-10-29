import sys
import os
import time
import socket
import webbrowser
import threading
import uvicorn

def is_port_in_use(port: int) -> bool:
    """Check if a network port is already in use on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find the first open port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    return start_port

def open_browser(port: int):
    """Wait briefly for the server to initialize, then open the browser."""
    time.sleep(1.2)
    url = f"http://127.0.0.1:{port}"
    print(f"\n[PORTAL LAUNCHED] Opening browser at {url} ...\n")
    webbrowser.open(url)

if __name__ == "__main__":
    print("=" * 65)
    print("  Canopy Vision - Automated Image Processing Portal")
    print("  Pipeline: RGB -> Depth Anything V2 + HSI + HSIE Fusion")
    print("=" * 65)
    
    # Smart port allocation
    selected_port = find_available_port(8000)
    if selected_port != 8000:
        print(f"[NOTE] Port 8000 was in use; automatically selected port {selected_port}.")
    
    # Launch browser in background thread
    threading.Thread(target=open_browser, args=(selected_port,), daemon=True).start()
    
    # Run Uvicorn server
    uvicorn.run("backend.app:app", host="127.0.0.1", port=selected_port, reload=False)

