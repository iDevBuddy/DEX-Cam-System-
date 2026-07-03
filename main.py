"""DEX AI Monitoring Demo — entry point.

Run:  python main.py
Then open http://localhost:8000 in your browser.
"""
import uvicorn

if __name__ == "__main__":
    print()
    print("  DEX AI — Worker Monitoring Demo")
    print("  Dashboard:  http://localhost:8000")
    print()
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, log_level="warning")
