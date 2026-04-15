#!/usr/bin/env python3
"""
Run the Biohydrogen Dashboard Web Server

Usage:
    python run_web.py

Then open your browser to: http://localhost:5000
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from web_app.app import app


if __name__ == '__main__':
    print("=" * 70)
    print("🌐 Biohydrogen Production Dashboard - Web Server Starting")
    print("=" * 70)
    print()
    print("📬 Dashboard URL: http://localhost:5000")
    print()
    print("Available Pages:")
    print("  • http://localhost:5000/              (Main Dashboard)")
    print("  • http://localhost:5000/sweep         (Parameter Sweep Analysis)")
    print("  • http://localhost:5000/cost          (Cost Analysis)")
    print("  • http://localhost:5000/detailed      (Detailed Results)")
    print()
    print("Press CTRL+C to stop the server")
    print("=" * 70)
    print()
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped.")
