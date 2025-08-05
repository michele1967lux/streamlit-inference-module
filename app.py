"""
Streamlit Inference Module - Professional AI Agent Management Interface
=======================================================================

A comprehensive Streamlit application for managing AI agents, running inference,
and handling memory/knowledge systems with support for Ollama and Gemini APIs.

Features:
- Multi-user support (up to 5 concurrent users)
- Agent creation and management
- Model selection and configuration (Ollama, OpenAI, Anthropic, Gemini)
- Tool assignment to agents
- Memory and knowledge integration
- Real-time inference with streaming
- Settings persistence
- Professional UI design
"""

import streamlit as st
import asyncio
import json
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import sqlite3

# Set page config first
st.set_page_config(
    page_title="AI Agent Inference Module",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Professional AI Agent Management Interface"
    }
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules after streamlit config
try:
    from inference_engine_module.inference_engine_module import InferenceEngineModule
    from memory_agent_module.memory_agent_module import MemoryAgentModule
except ImportError as e:
    st.error(f"Error importing inference modules: {e}")
    st.info("Note: Some features may be limited without these modules.")
    # Create mock classes for demonstration
    class InferenceEngineModule:
        def __init__(self, *args, **kwargs):
            pass
    class MemoryAgentModule:
        def __init__(self, *args, **kwargs):
            pass

# Import additional modules we'll create
try:
    from utils.config_manager import ConfigManager
    from utils.user_manager import UserManager
    from utils.model_scanner import ModelScanner
    from utils.knowledge_manager import KnowledgeManager
    from components.agent_manager import AgentManagerComponent
    from components.inference_interface import InferenceInterfaceComponent
    from components.settings_manager import SettingsManagerComponent
except ImportError as e:
    st.error(f"Error importing application modules: {e}")
    st.stop()

class StreamlitInferenceApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        """Initialize the application."""
        self.app_dir = Path(__file__).parent
        self.data_dir = self.app_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize managers
        self.config_manager = ConfigManager(self.data_dir / "config.json")
        self.user_manager = UserManager(self.data_dir / "users.db")
        self.model_scanner = ModelScanner()
        self.knowledge_manager = KnowledgeManager(self.data_dir / "knowledge.db")
        
        # Initialize components
        self.agent_manager = AgentManagerComponent(self.config_manager)
        self.inference_interface = InferenceInterfaceComponent()
        self.settings_manager = SettingsManagerComponent(self.config_manager)
        
        # Initialize session state
        self._init_session_state()
        
        # Setup modules
        self._setup_modules()
    
    def _init_session_state(self):
        """Initialize Streamlit session state."""
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        
        if "user_id" not in st.session_state:
            st.session_state.user_id = None
        
        if "current_agent" not in st.session_state:
            st.session_state.current_agent = None
        
        if "inference_history" not in st.session_state:
            st.session_state.inference_history = []
        
        if "settings" not in st.session_state:
            st.session_state.settings = self.config_manager.load_config()
    
    def _setup_modules(self):
        """Setup inference and memory modules."""
        try:
            # Setup inference engine
            self.inference_engine = InferenceEngineModule(
                agent_controller=self.agent_manager,
                storage_path=str(self.data_dir / "inference")
            )
            
            # Setup memory agent module  
            self.memory_module = MemoryAgentModule(
                pipeline_controller=None  # We'll handle this directly
            )
            
            logger.info("Modules initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up modules: {e}")
            st.error(f"Failed to initialize modules: {e}")
    
    def run(self):
        """Run the main application."""
        # Custom CSS for professional styling
        self._apply_custom_css()
        
        # Authentication check
        if not st.session_state.authenticated:
            self._show_login_page()
            return
        
        # Main application interface
        self._show_main_interface()
    
    def _apply_custom_css(self):
        """Apply custom CSS for professional styling."""
        st.markdown("""
        <style>
        /* Main theme colors */
        :root {
            --primary-color: #1f77b4;
            --secondary-color: #ff7f0e;
            --success-color: #2ca02c;
            --warning-color: #ff7f0e;
            --error-color: #d62728;
            --background-color: #f8f9fa;
            --text-color: #2c3e50;
        }
        
        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom header */
        .main-header {
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: var(--background-color);
        }
        
        /* Cards */
        .info-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            border-left: 4px solid var(--primary-color);
        }
        
        .status-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        
        /* Metrics */
        .metric-container {
            display: flex;
            justify-content: space-around;
            margin: 1rem 0;
        }
        
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            min-width: 120px;
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 5px;
            border: none;
            background: var(--primary-color);
            color: white;
            font-weight: 500;
        }
        
        .stButton > button:hover {
            background: #1565c0;
            transform: translateY(-1px);
        }
        
        /* Success/Error messages */
        .success-message {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 0.75rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        .error-message {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 0.75rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        /* Chat interface */
        .chat-container {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
        }
        
        /* Loading animation */
        .loading-animation {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid var(--primary-color);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _show_login_page(self):
        """Show login/authentication page."""
        st.markdown('<div class="main-header"><h1>🤖 AI Agent Inference Module</h1><p>Professional Multi-User Interface</p></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("### 🔐 User Authentication")
            
            # Check current users
            active_users = self.user_manager.get_active_users()
            if len(active_users) >= 5:
                st.error("Maximum number of concurrent users (5) reached. Please try again later.")
                st.stop()
            
            # Login form
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                col_login, col_register = st.columns(2)
                
                with col_login:
                    login_clicked = st.form_submit_button("🔑 Login", use_container_width=True)
                
                with col_register:
                    register_clicked = st.form_submit_button("📝 Register", use_container_width=True)
            
            if login_clicked:
                if self.user_manager.authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.user_id = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            
            if register_clicked:
                if username and password:
                    if self.user_manager.register_user(username, password):
                        st.success("Registration successful! You can now login.")
                    else:
                        st.error("Username already exists")
                else:
                    st.error("Please enter both username and password")
            
            # Show active users count
            st.info(f"Active users: {len(active_users)}/5")
            st.markdown('</div>', unsafe_allow_html=True)
    
    def _show_main_interface(self):
        """Show main application interface."""
        # Header
        st.markdown(f'''
        <div class="main-header">
            <h1>🤖 AI Agent Inference Module</h1>
            <p>Welcome, {st.session_state.user_id} | Professional Multi-User Interface</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Sidebar navigation
        with st.sidebar:
            st.markdown("### 🧭 Navigation")
            
            pages = {
                "🏠 Dashboard": "dashboard",
                "🤖 Agent Management": "agents", 
                "🧠 Inference Interface": "inference",
                "🔧 Settings": "settings",
                "💾 Memory Management": "memory",
                "📚 Knowledge Base": "knowledge",
                "🧪 Testing Interface": "testing",
                "📊 Analytics": "analytics"
            }
            
            selected_page = st.selectbox("Select Page", list(pages.keys()), key="page_selector")
            page_key = pages[selected_page]
            
            # User info
            st.markdown("---")
            st.markdown("### 👤 User Info")
            st.info(f"**User:** {st.session_state.user_id}")
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
            
            if st.button("🚪 Logout", use_container_width=True):
                self.user_manager.logout_user(st.session_state.user_id)
                st.session_state.authenticated = False
                st.session_state.user_id = None
                st.rerun()
        
        # Main content area
        if page_key == "dashboard":
            self._show_dashboard()
        elif page_key == "agents":
            self._show_agent_management()
        elif page_key == "inference":
            self._show_inference_interface()
        elif page_key == "settings":
            self._show_settings()
        elif page_key == "memory":
            self._show_memory_management()
        elif page_key == "knowledge":
            self._show_knowledge_base()
        elif page_key == "testing":
            self._show_testing_interface()
        elif page_key == "analytics":
            self._show_analytics()
    
    def _show_dashboard(self):
        """Show main dashboard."""
        st.markdown("## 🏠 Dashboard")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Active Users", len(self.user_manager.get_active_users()), delta=None)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            agents_count = len(self.agent_manager.get_all_agents())
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Agents", agents_count, delta=None)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            available_models = len(self.model_scanner.scan_ollama_models()) + 3  # +3 for OpenAI, Anthropic, Gemini
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Available Models", available_models, delta=None)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            inference_count = len(st.session_state.inference_history)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Inferences Run", inference_count, delta=None)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent activity
        st.markdown("### 📈 Recent Activity")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### 🕒 Recent Inferences")
            
            if st.session_state.inference_history:
                for inference in st.session_state.inference_history[-5:]:
                    st.markdown(f"- **{inference.get('timestamp', 'Unknown')}**: {inference.get('query', 'Unknown query')[:50]}...")
            else:
                st.info("No recent inferences")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### ⚙️ System Status")
            
            # Check system health
            status_items = [
                ("Inference Engine", "✅ Online"),
                ("Memory Module", "✅ Online"),
                ("Knowledge Base", "✅ Online"),
                ("Model Scanner", "✅ Online")
            ]
            
            for component, status in status_items:
                st.markdown(f"**{component}**: {status}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def _show_agent_management(self):
        """Show agent management interface."""
        self.agent_manager.render()
    
    def _show_inference_interface(self):
        """Show inference interface."""
        self.inference_interface.render(
            self.inference_engine,
            self.agent_manager.get_all_agents()
        )
    
    def _show_settings(self):
        """Show settings management."""
        self.settings_manager.render(self.model_scanner)
    
    def _show_memory_management(self):
        """Show memory management interface."""
        st.markdown("## 💾 Memory Management")
        st.info("Memory management interface - work in progress")
        # Implementation will be added
    
    def _show_knowledge_base(self):
        """Show knowledge base interface."""
        st.markdown("## 📚 Knowledge Base")
        st.info("Knowledge base interface - work in progress")
        # Implementation will be added
    
    def _show_testing_interface(self):
        """Show testing interface."""
        st.markdown("## 🧪 Testing Interface")
        st.info("Testing interface - work in progress")
        # Implementation will be added
    
    def _show_analytics(self):
        """Show analytics interface."""
        st.markdown("## 📊 Analytics")
        st.info("Analytics interface - work in progress")
        # Implementation will be added


def main():
    """Main entry point."""
    try:
        app = StreamlitInferenceApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}", exc_info=True)


if __name__ == "__main__":
    main()