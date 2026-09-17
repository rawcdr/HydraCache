import sys
import urllib.request
import urllib.error
import socket

def check_python_version():
    """Verify Python version is 3.10+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"OK Python Version: {sys.version}")
        return True
    else:
        print(f"FAIL Python Version: Expected 3.10+, got {sys.version}")
        return False

def check_qdrant(host='localhost', port=6333):
    """Verify Qdrant is running by checking the /readyz endpoint."""
    url = f"http://{host}:{port}/readyz"
    try:
        response = urllib.request.urlopen(url, timeout=5)
        if response.status == 200:
            print(f"OK Qdrant Connection: {url}")
            return True
        else:
            print(f"FAIL Qdrant Connection: Status {response.status} from {url}")
            return False
    except (urllib.error.URLError, ConnectionRefusedError) as e:
        print(f"FAIL Qdrant Connection: Cannot connect to {url}. Error: {e}")
        return False

def check_redis(host='localhost', port=6379):
    """Verify Redis is running by checking TCP port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    try:
        s.connect((host, port))
        print(f"OK Redis Connection: {host}:{port}")
        s.close()
        return True
    except (socket.error, ConnectionRefusedError) as e:
        print(f"FAIL Redis Connection: Cannot connect to {host}:{port}. Error: {e}")
        return False

if __name__ == "__main__":
    print("==================================================")
    print("HydraCache Infrastructure Verification")
    print("==================================================")
    
    python_ok = check_python_version()
    qdrant_ok = check_qdrant()
    redis_ok = check_redis()
    
    print("==================================================")
    if python_ok and qdrant_ok and redis_ok:
        print("🎉 All systems go! Phase 0 environment is ready.")
        sys.exit(0)
    else:
        print("⚠️  Some checks failed. Please review the output above.")
        sys.exit(1)
