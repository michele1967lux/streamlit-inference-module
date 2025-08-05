"""
Settings Manager Component for Streamlit Inference Module
=========================================================

Handles application settings, model configuration, and system prompts.
"""

import streamlit as st
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class SettingsManagerComponent:
    """Component for managing application settings."""
    
    def __init__(self, config_manager):
        """Initialize settings manager."""
        self.config_manager = config_manager
    
    def render(self, model_scanner):
        """Render settings management interface."""
        st.markdown("## 🔧 Settings")
        
        # Settings tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🤖 Model Configuration", 
            "📝 System Prompts", 
            "⚙️ General Settings",
            "🎨 UI Preferences",
            "🔒 Security & Users"
        ])
        
        with tab1:
            self._render_model_configuration(model_scanner)
        
        with tab2:
            self._render_system_prompts()
        
        with tab3:
            self._render_general_settings()
        
        with tab4:
            self._render_ui_preferences()
        
        with tab5:
            self._render_security_settings()
    
    def _render_model_configuration(self, model_scanner):
        """Render model configuration interface."""
        st.markdown("### 🤖 Model Configuration")
        
        # Model provider tabs
        provider_tabs = st.tabs(["🦙 Ollama", "🤖 OpenAI", "🧠 Anthropic", "💎 Gemini"])
        
        # Ollama Configuration
        with provider_tabs[0]:
            st.markdown("#### 🦙 Ollama Configuration")
            
            # Ollama status check
            ollama_config = self.config_manager.get_model_config("ollama")
            base_url = st.text_input(
                "Ollama Base URL",
                value=ollama_config.get("base_url", "http://localhost:11434"),
                help="URL where Ollama service is running"
            )
            
            # Test connection
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Test Connection", key="test_ollama"):
                    status = model_scanner.check_ollama_status(base_url)
                    
                    if status["status"] == "online":
                        st.success(f"✅ Connected! Version: {status['version']}")
                    else:
                        st.error(f"❌ {status['message']}")
            
            with col2:
                if st.button("🔄 Scan Models", key="scan_ollama"):
                    with st.spinner("Scanning Ollama models..."):
                        models = model_scanner.scan_ollama_models(base_url)
                        
                        if models:
                            st.success(f"✅ Found {len(models)} models")
                            # Update config with available models
                            self.config_manager.set("models.ollama.available_models", 
                                                   [m["name"] for m in models])
                        else:
                            st.warning("No models found")
            
            # Available models
            available_models = model_scanner.scan_ollama_models(base_url)
            
            if available_models:
                st.markdown("##### 📋 Available Models")
                
                for model in available_models:
                    with st.expander(f"🤖 {model['name']}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.text(f"Size: {self._format_size(model.get('size', 0))}")
                        
                        with col2:
                            st.text(f"Modified: {model.get('modified_at', 'Unknown')[:10]}")
                        
                        with col3:
                            if st.button(f"🗑️ Remove", key=f"remove_{model['name']}"):
                                result = model_scanner.remove_ollama_model(model['name'], base_url)
                                if result["success"]:
                                    st.success("Model removed!")
                                    st.rerun()
                                else:
                                    st.error(f"Failed: {result['message']}")
                        
                        # Model details
                        if model.get('details'):
                            st.json(model['details'])
            else:
                st.info("No models available. Install some models below.")
            
            # Model installation
            st.markdown("##### ➕ Install New Models")
            
            install_tab1, install_tab2 = st.tabs(["🌟 Popular Models", "🔍 Custom Model"])
            
            with install_tab1:
                popular_models = model_scanner.get_popular_models()
                
                for model in popular_models:
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{model['name']}**")
                            st.caption(f"{model['description']} | Size: {model['size']}")
                            
                            if model.get('recommended'):
                                st.success("⭐ Recommended")
                        
                        with col2:
                            st.text(f"Category: {model['category']}")
                        
                        with col3:
                            if st.button(f"📥 Install", key=f"install_{model['name']}"):
                                with st.spinner(f"Installing {model['name']}..."):
                                    result = model_scanner.install_ollama_model(model['name'], base_url)
                                    
                                    if result["success"]:
                                        st.success(f"✅ {model['name']} installation started!")
                                    else:
                                        st.error(f"❌ {result['message']}")
                        
                        st.divider()
            
            with install_tab2:
                with st.form("install_custom_model"):
                    custom_model = st.text_input(
                        "Model Name",
                        placeholder="e.g., llama3.1:8b, mistral:7b, codellama:13b"
                    )
                    
                    if st.form_submit_button("📥 Install Custom Model"):
                        if custom_model:
                            with st.spinner(f"Installing {custom_model}..."):
                                result = model_scanner.install_ollama_model(custom_model, base_url)
                                
                                if result["success"]:
                                    st.success(f"✅ {custom_model} installation started!")
                                else:
                                    st.error(f"❌ {result['message']}")
                        else:
                            st.error("Please enter a model name")
            
            # Save Ollama configuration
            if st.button("💾 Save Ollama Settings", key="save_ollama"):
                ollama_settings = {
                    "enabled": True,
                    "base_url": base_url,
                    "available_models": [m["name"] for m in available_models]
                }
                
                if self.config_manager.set_model_config("ollama", ollama_settings):
                    st.success("✅ Ollama settings saved!")
                else:
                    st.error("❌ Failed to save settings")
        
        # OpenAI Configuration
        with provider_tabs[1]:
            st.markdown("#### 🤖 OpenAI Configuration")
            
            openai_config = self.config_manager.get_model_config("openai")
            
            with st.form("openai_config"):
                enabled = st.checkbox("Enable OpenAI", value=openai_config.get("enabled", False))
                
                api_key = st.text_input(
                    "API Key",
                    value=openai_config.get("api_key", ""),
                    type="password",
                    help="Your OpenAI API key"
                )
                
                base_url = st.text_input(
                    "Base URL (optional)",
                    value=openai_config.get("base_url", "https://api.openai.com/v1"),
                    help="Custom base URL for OpenAI-compatible APIs"
                )
                
                # Model selection
                openai_models = model_scanner.get_openai_models()
                model_options = [m["name"] for m in openai_models]
                
                current_model = openai_config.get("model", "gpt-4o")
                if current_model in model_options:
                    model_index = model_options.index(current_model)
                else:
                    model_index = 0
                
                selected_model = st.selectbox(
                    "Default Model",
                    model_options,
                    index=model_index
                )
                
                # Test API key
                test_api = st.checkbox("Test API Key", value=False)
                
                if st.form_submit_button("💾 Save OpenAI Settings"):
                    openai_settings = {
                        "enabled": enabled,
                        "api_key": api_key,
                        "base_url": base_url,
                        "model": selected_model
                    }
                    
                    if self.config_manager.set_model_config("openai", openai_settings):
                        st.success("✅ OpenAI settings saved!")
                        
                        if test_api and api_key:
                            st.info("🧪 API key testing not implemented yet")
                    else:
                        st.error("❌ Failed to save settings")
            
            # Display available models
            st.markdown("##### 📋 Available OpenAI Models")
            for model in openai_models:
                with st.expander(f"🤖 {model['name']}"):
                    st.text(f"Description: {model['description']}")
                    st.text(f"Context Length: {model.get('context_length', 'Unknown'):,}")
                    if model.get('multimodal'):
                        st.success("🎨 Supports multimodal input")
        
        # Anthropic Configuration
        with provider_tabs[2]:
            st.markdown("#### 🧠 Anthropic Configuration")
            
            anthropic_config = self.config_manager.get_model_config("anthropic")
            
            with st.form("anthropic_config"):
                enabled = st.checkbox("Enable Anthropic", value=anthropic_config.get("enabled", False))
                
                api_key = st.text_input(
                    "API Key",
                    value=anthropic_config.get("api_key", ""),
                    type="password",
                    help="Your Anthropic API key"
                )
                
                # Model selection
                anthropic_models = model_scanner.get_anthropic_models()
                model_options = [m["name"] for m in anthropic_models]
                
                current_model = anthropic_config.get("model", "claude-3-5-sonnet-20240620")
                if current_model in model_options:
                    model_index = model_options.index(current_model)
                else:
                    model_index = 0
                
                selected_model = st.selectbox(
                    "Default Model",
                    model_options,
                    index=model_index
                )
                
                if st.form_submit_button("💾 Save Anthropic Settings"):
                    anthropic_settings = {
                        "enabled": enabled,
                        "api_key": api_key,
                        "model": selected_model
                    }
                    
                    if self.config_manager.set_model_config("anthropic", anthropic_settings):
                        st.success("✅ Anthropic settings saved!")
                    else:
                        st.error("❌ Failed to save settings")
            
            # Display available models
            st.markdown("##### 📋 Available Anthropic Models")
            for model in anthropic_models:
                with st.expander(f"🧠 {model['name']}"):
                    st.text(f"Description: {model['description']}")
                    st.text(f"Context Length: {model.get('context_length', 'Unknown'):,}")
        
        # Gemini Configuration
        with provider_tabs[3]:
            st.markdown("#### 💎 Gemini Configuration")
            
            gemini_config = self.config_manager.get_model_config("gemini")
            
            with st.form("gemini_config"):
                enabled = st.checkbox("Enable Gemini", value=gemini_config.get("enabled", False))
                
                api_key = st.text_input(
                    "API Key",
                    value=gemini_config.get("api_key", ""),
                    type="password",
                    help="Your Google AI API key"
                )
                
                # Model selection
                gemini_models = model_scanner.get_gemini_models()
                model_options = [m["name"] for m in gemini_models]
                
                current_model = gemini_config.get("model", "gemini-1.5-pro")
                if current_model in model_options:
                    model_index = model_options.index(current_model)
                else:
                    model_index = 0
                
                selected_model = st.selectbox(
                    "Default Model",
                    model_options,
                    index=model_index
                )
                
                if st.form_submit_button("💾 Save Gemini Settings"):
                    gemini_settings = {
                        "enabled": enabled,
                        "api_key": api_key,
                        "model": selected_model
                    }
                    
                    if self.config_manager.set_model_config("gemini", gemini_settings):
                        st.success("✅ Gemini settings saved!")
                    else:
                        st.error("❌ Failed to save settings")
            
            # Display available models
            st.markdown("##### 📋 Available Gemini Models")
            for model in gemini_models:
                with st.expander(f"💎 {model['name']}"):
                    st.text(f"Description: {model['description']}")
                    st.text(f"Context Length: {model.get('context_length', 'Unknown'):,}")
                    if model.get('multimodal'):
                        st.success("🎨 Supports multimodal input")
    
    def _render_system_prompts(self):
        """Render system prompts management."""
        st.markdown("### 📝 System Prompts")
        
        prompt_tabs = st.tabs(["📋 View Prompts", "➕ Create Custom", "⚙️ Manage Prompts"])
        
        # View existing prompts
        with prompt_tabs[0]:
            st.markdown("#### 📋 Available System Prompts")
            
            all_prompts = self.config_manager.get_all_prompts()
            
            for name, prompt in all_prompts.items():
                with st.expander(f"📝 {name.title()}"):
                    st.markdown("**Prompt:**")
                    st.text_area("", value=prompt, height=100, disabled=True, key=f"view_{name}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text(f"Length: {len(prompt)} characters")
                    with col2:
                        if name not in ["default", "creative", "analytical"]:  # Custom prompts
                            if st.button(f"🗑️ Delete", key=f"delete_prompt_{name}"):
                                if self.config_manager.delete_custom_prompt(name):
                                    st.success(f"Prompt '{name}' deleted!")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete prompt")
        
        # Create custom prompt
        with prompt_tabs[1]:
            st.markdown("#### ➕ Create Custom System Prompt")
            
            with st.form("create_custom_prompt"):
                prompt_name = st.text_input(
                    "Prompt Name",
                    placeholder="e.g., technical_writer, customer_support"
                )
                
                prompt_content = st.text_area(
                    "Prompt Content",
                    height=200,
                    placeholder="Enter your custom system prompt here..."
                )
                
                # Prompt templates
                st.markdown("##### 🔖 Quick Templates")
                
                template_col1, template_col2 = st.columns(2)
                
                with template_col1:
                    if st.button("💼 Professional Assistant"):
                        st.session_state.prompt_template = """You are a professional AI assistant. You provide:
- Clear, concise, and accurate information
- Well-structured responses with proper formatting
- Professional tone while remaining approachable
- Evidence-based answers when possible
- Helpful suggestions and next steps

Always maintain professionalism while being helpful and engaging."""
                
                with template_col2:
                    if st.button("🎨 Creative Writer"):
                        st.session_state.prompt_template = """You are a creative writing assistant. You help with:
- Creative story development and plot ideas
- Character development and dialogue
- Writing style and tone guidance
- Overcoming writer's block
- Editing and improving existing content

Be imaginative, inspiring, and supportive while providing constructive feedback."""
                
                # Use template if selected
                if hasattr(st.session_state, 'prompt_template'):
                    prompt_content = st.text_area(
                        "Prompt Content (with template)",
                        value=st.session_state.prompt_template,
                        height=200
                    )
                    del st.session_state.prompt_template
                
                if st.form_submit_button("💾 Save Custom Prompt"):
                    if prompt_name and prompt_content:
                        if self.config_manager.save_custom_prompt(prompt_name, prompt_content):
                            st.success(f"✅ Custom prompt '{prompt_name}' saved!")
                        else:
                            st.error("❌ Failed to save prompt")
                    else:
                        st.error("Please enter both name and content")
        
        # Manage existing prompts
        with prompt_tabs[2]:
            st.markdown("#### ⚙️ Manage System Prompts")
            
            # Default prompts configuration
            st.markdown("##### 🔧 Default Prompts")
            
            default_prompts = {
                "default": "Default Assistant",
                "creative": "Creative Assistant", 
                "analytical": "Analytical Assistant"
            }
            
            for key, label in default_prompts.items():
                with st.expander(f"✏️ Edit {label}"):
                    current_prompt = self.config_manager.get_system_prompt(key)
                    
                    new_prompt = st.text_area(
                        f"{label} Prompt",
                        value=current_prompt,
                        height=100,
                        key=f"edit_{key}"
                    )
                    
                    if st.button(f"💾 Update {label}", key=f"update_{key}"):
                        if self.config_manager.set(f"system_prompts.{key}", new_prompt):
                            st.success(f"✅ {label} prompt updated!")
                        else:
                            st.error("❌ Failed to update prompt")
            
            # Import/Export prompts
            st.markdown("##### 📤 Import/Export")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📥 Import Prompts**")
                uploaded_file = st.file_uploader(
                    "Upload prompt file",
                    type=['json'],
                    help="Upload a JSON file with custom prompts"
                )
                
                if uploaded_file:
                    try:
                        import json
                        prompts_data = json.load(uploaded_file)
                        
                        if isinstance(prompts_data, dict):
                            imported_count = 0
                            for name, prompt in prompts_data.items():
                                if self.config_manager.save_custom_prompt(name, prompt):
                                    imported_count += 1
                            
                            st.success(f"✅ Imported {imported_count} prompts!")
                        else:
                            st.error("Invalid file format")
                    except Exception as e:
                        st.error(f"Import failed: {str(e)}")
            
            with col2:
                st.markdown("**📤 Export Prompts**")
                
                if st.button("📋 Export All Prompts"):
                    try:
                        import json
                        all_prompts = self.config_manager.get_all_prompts()
                        
                        st.download_button(
                            "📥 Download Prompts",
                            data=json.dumps(all_prompts, indent=2),
                            file_name="system_prompts.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
    
    def _render_general_settings(self):
        """Render general application settings."""
        st.markdown("### ⚙️ General Settings")
        
        # Inference settings
        st.markdown("#### 🧠 Inference Defaults")
        
        inference_config = self.config_manager.get("inference", {})
        
        with st.form("inference_settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                default_stream = st.checkbox(
                    "Enable Streaming by Default",
                    value=inference_config.get("default_stream", True)
                )
                
                show_tool_calls = st.checkbox(
                    "Show Tool Calls by Default",
                    value=inference_config.get("show_tool_calls", True)
                )
                
                cache_responses = st.checkbox(
                    "Cache Responses",
                    value=inference_config.get("cache_responses", True)
                )
            
            with col2:
                default_temperature = st.slider(
                    "Default Temperature",
                    0.0, 2.0,
                    inference_config.get("default_temperature", 0.7),
                    0.1
                )
                
                default_max_tokens = st.number_input(
                    "Default Max Tokens",
                    100, 8192,
                    inference_config.get("default_max_tokens", 2048)
                )
                
                show_reasoning = st.checkbox(
                    "Show Reasoning by Default",
                    value=inference_config.get("show_reasoning", False)
                )
            
            if st.form_submit_button("💾 Save Inference Settings"):
                inference_settings = {
                    "default_stream": default_stream,
                    "default_temperature": default_temperature,
                    "default_max_tokens": default_max_tokens,
                    "show_tool_calls": show_tool_calls,
                    "show_reasoning": show_reasoning,
                    "cache_responses": cache_responses,
                    "save_history": True
                }
                
                if self.config_manager.update_section("inference", inference_settings):
                    st.success("✅ Inference settings saved!")
                else:
                    st.error("❌ Failed to save settings")
        
        # Memory settings
        st.markdown("#### 🧠 Memory Settings")
        
        memory_config = self.config_manager.get("memory", {})
        
        with st.form("memory_settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                memory_enabled = st.checkbox(
                    "Enable Memory",
                    value=memory_config.get("enabled", True)
                )
                
                auto_summarize = st.checkbox(
                    "Auto Summarize Sessions",
                    value=memory_config.get("auto_summarize", True)
                )
            
            with col2:
                max_memories = st.number_input(
                    "Max Memories per User",
                    100, 10000,
                    memory_config.get("max_memories_per_user", 1000)
                )
                
                retention_days = st.number_input(
                    "Memory Retention (days)",
                    1, 365,
                    memory_config.get("retention_days", 30)
                )
            
            if st.form_submit_button("💾 Save Memory Settings"):
                memory_settings = {
                    "enabled": memory_enabled,
                    "max_memories_per_user": max_memories,
                    "auto_summarize": auto_summarize,
                    "retention_days": retention_days
                }
                
                if self.config_manager.update_section("memory", memory_settings):
                    st.success("✅ Memory settings saved!")
                else:
                    st.error("❌ Failed to save settings")
        
        # Knowledge settings
        st.markdown("#### 📚 Knowledge Settings")
        
        knowledge_config = self.config_manager.get("knowledge", {})
        
        with st.form("knowledge_settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                knowledge_enabled = st.checkbox(
                    "Enable Knowledge Base",
                    value=knowledge_config.get("enabled", True)
                )
                
                max_documents = st.number_input(
                    "Max Documents per KB",
                    10, 1000,
                    knowledge_config.get("max_documents", 100)
                )
            
            with col2:
                chunk_size = st.number_input(
                    "Document Chunk Size",
                    500, 2000,
                    knowledge_config.get("chunk_size", 1000)
                )
                
                chunk_overlap = st.number_input(
                    "Chunk Overlap",
                    50, 500,
                    knowledge_config.get("overlap", 200)
                )
            
            if st.form_submit_button("💾 Save Knowledge Settings"):
                knowledge_settings = {
                    "enabled": knowledge_enabled,
                    "max_documents": max_documents,
                    "chunk_size": chunk_size,
                    "overlap": chunk_overlap
                }
                
                if self.config_manager.update_section("knowledge", knowledge_settings):
                    st.success("✅ Knowledge settings saved!")
                else:
                    st.error("❌ Failed to save settings")
    
    def _render_ui_preferences(self):
        """Render UI preferences."""
        st.markdown("### 🎨 UI Preferences")
        
        ui_config = self.config_manager.get("ui", {})
        
        with st.form("ui_preferences"):
            col1, col2 = st.columns(2)
            
            with col1:
                theme = st.selectbox(
                    "Theme",
                    ["professional", "dark", "light"],
                    index=["professional", "dark", "light"].index(ui_config.get("theme", "professional"))
                )
                
                sidebar_expanded = st.checkbox(
                    "Sidebar Expanded by Default",
                    value=ui_config.get("sidebar_expanded", True)
                )
            
            with col2:
                show_advanced = st.checkbox(
                    "Show Advanced Options",
                    value=ui_config.get("show_advanced_options", False)
                )
                
                auto_refresh = st.checkbox(
                    "Auto Refresh",
                    value=ui_config.get("auto_refresh", True)
                )
            
            if st.form_submit_button("💾 Save UI Preferences"):
                ui_settings = {
                    "theme": theme,
                    "sidebar_expanded": sidebar_expanded,
                    "show_advanced_options": show_advanced,
                    "auto_refresh": auto_refresh
                }
                
                if self.config_manager.update_section("ui", ui_settings):
                    st.success("✅ UI preferences saved!")
                    st.info("Some changes may require a page refresh to take effect.")
                else:
                    st.error("❌ Failed to save preferences")
    
    def _render_security_settings(self):
        """Render security and user settings."""
        st.markdown("### 🔒 Security & User Management")
        
        app_config = self.config_manager.get("app", {})
        
        # App settings
        st.markdown("#### 🛡️ Application Security")
        
        with st.form("security_settings"):
            max_users = st.number_input(
                "Maximum Concurrent Users",
                1, 20,
                app_config.get("max_users", 5)
            )
            
            st.info("Current session timeout: 24 hours (hardcoded)")
            
            if st.form_submit_button("💾 Save Security Settings"):
                app_settings = {
                    **app_config,
                    "max_users": max_users
                }
                
                if self.config_manager.update_section("app", app_settings):
                    st.success("✅ Security settings saved!")
                else:
                    st.error("❌ Failed to save settings")
        
        # Configuration management
        st.markdown("#### 📋 Configuration Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                if st.button("⚠️ Confirm Reset", key="confirm_reset", type="secondary"):
                    if self.config_manager.reset_to_defaults():
                        st.success("✅ Configuration reset to defaults!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to reset configuration")
        
        with col2:
            if st.button("📤 Export Config", use_container_width=True):
                try:
                    import json
                    from pathlib import Path
                    
                    config_data = json.dumps(self.config_manager.config, indent=2)
                    
                    st.download_button(
                        "📥 Download Config",
                        data=config_data,
                        file_name="app_config.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
        
        with col3:
            uploaded_config = st.file_uploader(
                "📥 Import Config",
                type=['json'],
                help="Upload a configuration file"
            )
            
            if uploaded_config:
                try:
                    import json
                    import tempfile
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        config_data = json.load(uploaded_config)
                        json.dump(config_data, f)
                        temp_path = f.name
                    
                    if self.config_manager.import_config(Path(temp_path)):
                        st.success("✅ Configuration imported!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to import configuration")
                        
                except Exception as e:
                    st.error(f"Import failed: {str(e)}")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format byte size to human readable string."""
        try:
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size_bytes < 1024.0:
                    return f"{size_bytes:.1f} {unit}"
                size_bytes /= 1024.0
            return f"{size_bytes:.1f} PB"
        except:
            return "Unknown"