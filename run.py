#!/usr/bin/env python3
"""
Streamlit Inference Module - Startup Script
===========================================

This script starts the Streamlit application for AI agent inference and management.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def main():
    """Main entry point for the application."""
    try:
        # Check if streamlit is installed
        try:
            import streamlit
        except ImportError:
            print("❌ Streamlit not found. Please install requirements first:")
            print("pip install -r requirements.txt")
            sys.exit(1)
        
        # Set environment variables for better Streamlit experience
        os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
        os.environ.setdefault("STREAMLIT_SERVER_ENABLE_CORS", "false")
        os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")
        
        print("🚀 Starting Streamlit Inference Module...")
        print("📍 Navigate to: http://localhost:8501")
        print("💡 Use Ctrl+C to stop the application")
        
        # Run streamlit
        from streamlit.web.cli import main as streamlit_main
        sys.argv = ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
        streamlit_main()
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()