"""
Agent Manager Component for Streamlit Inference Module
======================================================

Handles agent creation, configuration, and management interface.
"""

import streamlit as st
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class AgentManagerComponent:
    """Component for managing AI agents."""
    
    def __init__(self, config_manager):
        """Initialize agent manager."""
        self.config_manager = config_manager
        self.agents_config_key = "agents"
        
        # Initialize agents in config if not exists
        if not self.config_manager.get(self.agents_config_key):
            self.config_manager.set(self.agents_config_key, {})
    
    def render(self):
        """Render agent management interface."""
        st.markdown("## 🤖 Agent Management")
        
        # Tabs for different agent operations
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Agent List", "➕ Create Agent", "⚙️ Configure Agent", "🔧 Tools & Knowledge"])
        
        with tab1:
            self._render_agent_list()
        
        with tab2:
            self._render_create_agent()
        
        with tab3:
            self._render_configure_agent()
        
        with tab4:
            self._render_tools_knowledge()
    
    def _render_agent_list(self):
        """Render list of existing agents."""
        st.markdown("### 📋 Active Agents")
        
        agents = self.get_all_agents()
        
        if not agents:
            st.info("No agents created yet. Use the 'Create Agent' tab to get started.")
            return
        
        # Agent cards
        for agent_id, agent_data in agents.items():
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    st.markdown(f"**{agent_data.get('name', 'Unnamed Agent')}**")
                    st.caption(agent_data.get('description', 'No description'))
                
                with col2:
                    model_info = agent_data.get('model', {})
                    st.text(f"Provider: {model_info.get('provider', 'Unknown')}")
                    st.text(f"Model: {model_info.get('model_name', 'Unknown')}")
                
                with col3:
                    tools_count = len(agent_data.get('tools', []))
                    knowledge_count = len(agent_data.get('knowledge_bases', []))
                    st.text(f"Tools: {tools_count}")
                    st.text(f"Knowledge: {knowledge_count}")
                
                with col4:
                    if st.button(f"🗑️", key=f"delete_{agent_id}", help="Delete Agent"):
                        self.delete_agent(agent_id)
                        st.rerun()
                
                # Agent status
                status = "🟢 Active" if agent_data.get('active', True) else "🔴 Inactive"
                st.caption(f"Status: {status} | Created: {agent_data.get('created_at', 'Unknown')}")
                
                st.divider()
    
    def _render_create_agent(self):
        """Render agent creation form."""
        st.markdown("### ➕ Create New Agent")
        
        with st.form("create_agent_form"):
            # Basic information
            st.markdown("#### Basic Information")
            name = st.text_input("Agent Name*", placeholder="Enter agent name")
            description = st.text_area("Description", placeholder="Describe what this agent does")
            
            # Model selection
            st.markdown("#### Model Configuration")
            
            # Provider selection
            providers = ["ollama", "openai", "anthropic", "gemini"]
            provider = st.selectbox("Model Provider*", providers)
            
            # Model selection based on provider
            if provider == "ollama":
                from utils.model_scanner import ModelScanner
                scanner = ModelScanner()
                ollama_models = scanner.scan_ollama_models()
                
                if ollama_models:
                    model_names = [m["name"] for m in ollama_models]
                    model_name = st.selectbox("Ollama Model*", model_names)
                else:
                    st.warning("No Ollama models found. Make sure Ollama is running and has models installed.")
                    model_name = st.text_input("Model Name*", placeholder="e.g., llama3.1:8b")
            
            elif provider == "openai":
                openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
                model_name = st.selectbox("OpenAI Model*", openai_models)
            
            elif provider == "anthropic":
                anthropic_models = [
                    "claude-3-5-sonnet-20240620",
                    "claude-3-opus-20240229", 
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307"
                ]
                model_name = st.selectbox("Anthropic Model*", anthropic_models)
            
            elif provider == "gemini":
                gemini_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
                model_name = st.selectbox("Gemini Model*", gemini_models)
            
            # System prompt
            st.markdown("#### System Prompt")
            
            # Predefined prompts
            all_prompts = self.config_manager.get_all_prompts()
            prompt_names = list(all_prompts.keys())
            
            col1, col2 = st.columns([1, 1])
            with col1:
                selected_prompt = st.selectbox("Use Predefined Prompt", ["custom"] + prompt_names)
            
            with col2:
                if st.button("👁️ Preview Prompt") and selected_prompt != "custom":
                    st.info(all_prompts.get(selected_prompt, ""))
            
            if selected_prompt == "custom":
                system_prompt = st.text_area(
                    "Custom System Prompt*",
                    value="You are a helpful AI assistant. Provide clear, accurate, and helpful responses.",
                    height=100
                )
            else:
                system_prompt = all_prompts.get(selected_prompt, "")
                st.info(f"Using prompt: {selected_prompt}")
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                col1, col2 = st.columns(2)
                
                with col1:
                    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
                    max_tokens = st.number_input("Max Tokens", 100, 8192, 2048)
                
                with col2:
                    enable_memory = st.checkbox("Enable Memory", value=True)
                    enable_tools = st.checkbox("Enable Tools", value=True)
            
            # Submit button
            submitted = st.form_submit_button("🚀 Create Agent", use_container_width=True)
            
            if submitted:
                if not name or not model_name or not system_prompt:
                    st.error("Please fill in all required fields (marked with *)")
                else:
                    success = self.create_agent(
                        name=name,
                        description=description,
                        provider=provider,
                        model_name=model_name,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        enable_memory=enable_memory,
                        enable_tools=enable_tools
                    )
                    
                    if success:
                        st.success(f"✅ Agent '{name}' created successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to create agent. Please check the logs.")
    
    def _render_configure_agent(self):
        """Render agent configuration interface."""
        st.markdown("### ⚙️ Configure Existing Agent")
        
        agents = self.get_all_agents()
        
        if not agents:
            st.info("No agents available to configure.")
            return
        
        # Agent selection
        agent_names = {f"{data['name']} ({agent_id})": agent_id for agent_id, data in agents.items()}
        selected_agent = st.selectbox("Select Agent to Configure", list(agent_names.keys()))
        
        if selected_agent:
            agent_id = agent_names[selected_agent]
            agent_data = agents[agent_id]
            
            # Configuration form
            with st.form(f"config_agent_{agent_id}"):
                # Basic settings
                st.markdown("#### Basic Settings")
                new_name = st.text_input("Name", value=agent_data.get('name', ''))
                new_description = st.text_area("Description", value=agent_data.get('description', ''))
                
                # Model settings
                st.markdown("#### Model Settings")
                current_model = agent_data.get('model', {})
                
                col1, col2 = st.columns(2)
                with col1:
                    new_temperature = st.slider("Temperature", 0.0, 2.0, 
                                               current_model.get('temperature', 0.7), 0.1)
                with col2:
                    new_max_tokens = st.number_input("Max Tokens", 100, 8192, 
                                                   current_model.get('max_tokens', 2048))
                
                # System prompt
                st.markdown("#### System Prompt")
                current_prompt = agent_data.get('system_prompt', '')
                new_system_prompt = st.text_area("System Prompt", value=current_prompt, height=100)
                
                # Status
                st.markdown("#### Status")
                new_active = st.checkbox("Active", value=agent_data.get('active', True))
                
                # Submit button
                if st.form_submit_button("💾 Update Configuration", use_container_width=True):
                    updates = {
                        'name': new_name,
                        'description': new_description,
                        'system_prompt': new_system_prompt,
                        'active': new_active,
                        'model': {
                            **current_model,
                            'temperature': new_temperature,
                            'max_tokens': new_max_tokens
                        },
                        'updated_at': datetime.now().isoformat()
                    }
                    
                    if self.update_agent(agent_id, updates):
                        st.success("✅ Agent configuration updated!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to update agent configuration.")
    
    def _render_tools_knowledge(self):
        """Render tools and knowledge assignment interface."""
        st.markdown("### 🔧 Tools & Knowledge Assignment")
        
        agents = self.get_all_agents()
        
        if not agents:
            st.info("No agents available.")
            return
        
        # Agent selection
        agent_names = {f"{data['name']} ({agent_id})": agent_id for agent_id, data in agents.items()}
        selected_agent = st.selectbox("Select Agent", list(agent_names.keys()), key="tools_agent")
        
        if selected_agent:
            agent_id = agent_names[selected_agent]
            agent_data = agents[agent_id]
            
            col1, col2 = st.columns(2)
            
            # Tools assignment
            with col1:
                st.markdown("#### 🛠️ Available Tools")
                
                available_tools = self._get_available_tools()
                current_tools = agent_data.get('tools', [])
                
                with st.form(f"tools_{agent_id}"):
                    new_tools = []
                    
                    for tool_id, tool_info in available_tools.items():
                        is_assigned = tool_id in current_tools
                        if st.checkbox(f"{tool_info['name']}", value=is_assigned, key=f"tool_{tool_id}"):
                            new_tools.append(tool_id)
                        st.caption(tool_info['description'])
                    
                    if st.form_submit_button("🔄 Update Tools"):
                        self.update_agent_tools(agent_id, new_tools)
                        st.success("Tools updated!")
                        st.rerun()
            
            # Knowledge base assignment
            with col2:
                st.markdown("#### 📚 Knowledge Bases")
                
                # Get available knowledge bases
                from utils.knowledge_manager import KnowledgeManager
                knowledge_manager = KnowledgeManager(self.config_manager.config_path.parent / "knowledge.db")
                available_kb = knowledge_manager.get_knowledge_bases()
                current_kb = agent_data.get('knowledge_bases', [])
                
                if available_kb:
                    with st.form(f"knowledge_{agent_id}"):
                        new_kb = []
                        
                        for kb in available_kb:
                            is_assigned = kb['id'] in current_kb
                            if st.checkbox(f"{kb['name']}", value=is_assigned, key=f"kb_{kb['id']}"):
                                new_kb.append(kb['id'])
                            st.caption(f"{kb['description']} ({kb['document_count']} docs)")
                        
                        if st.form_submit_button("🔄 Update Knowledge"):
                            self.update_agent_knowledge(agent_id, new_kb)
                            st.success("Knowledge bases updated!")
                            st.rerun()
                else:
                    st.info("No knowledge bases available. Create one in the Knowledge Base section.")
    
    def _get_available_tools(self) -> Dict[str, Dict[str, str]]:
        """Get available tools for assignment."""
        return {
            "web_search": {
                "name": "🔍 Web Search",
                "description": "Search the web for current information"
            },
            "calculator": {
                "name": "🧮 Calculator",
                "description": "Perform mathematical calculations"
            },
            "file_reader": {
                "name": "📄 File Reader",
                "description": "Read and analyze files"
            },
            "code_executor": {
                "name": "💻 Code Executor",
                "description": "Execute Python code safely"
            },
            "reasoning": {
                "name": "🧠 Reasoning Tools",
                "description": "Advanced reasoning and logic tools"
            },
            "memory_search": {
                "name": "🧠 Memory Search",
                "description": "Search through conversation memory"
            }
        }
    
    def create_agent(self, name: str, description: str, provider: str, model_name: str,
                    system_prompt: str, temperature: float = 0.7, max_tokens: int = 2048,
                    enable_memory: bool = True, enable_tools: bool = True) -> bool:
        """Create a new agent."""
        try:
            agent_id = str(uuid.uuid4())
            
            agent_data = {
                "id": agent_id,
                "name": name,
                "description": description,
                "model": {
                    "provider": provider,
                    "model_name": model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                "system_prompt": system_prompt,
                "enable_memory": enable_memory,
                "enable_tools": enable_tools,
                "tools": [],
                "knowledge_bases": [],
                "active": True,
                "created_at": datetime.now().isoformat(),
                "created_by": st.session_state.get('user_id', 'unknown')
            }
            
            # Update config
            agents = self.config_manager.get(self.agents_config_key, {})
            agents[agent_id] = agent_data
            
            return self.config_manager.set(self.agents_config_key, agents)
            
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            return False
    
    def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing agent."""
        try:
            agents = self.config_manager.get(self.agents_config_key, {})
            
            if agent_id in agents:
                agents[agent_id].update(updates)
                return self.config_manager.set(self.agents_config_key, agents)
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating agent: {e}")
            return False
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        try:
            agents = self.config_manager.get(self.agents_config_key, {})
            
            if agent_id in agents:
                del agents[agent_id]
                return self.config_manager.set(self.agents_config_key, agents)
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting agent: {e}")
            return False
    
    def get_all_agents(self) -> Dict[str, Any]:
        """Get all agents."""
        return self.config_manager.get(self.agents_config_key, {})
    
    def get_agent_by_id(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent by ID."""
        agents = self.get_all_agents()
        return agents.get(agent_id)
    
    def update_agent_tools(self, agent_id: str, tools: List[str]) -> bool:
        """Update agent tools."""
        return self.update_agent(agent_id, {"tools": tools})
    
    def update_agent_knowledge(self, agent_id: str, knowledge_bases: List[str]) -> bool:
        """Update agent knowledge bases."""
        return self.update_agent(agent_id, {"knowledge_bases": knowledge_bases})