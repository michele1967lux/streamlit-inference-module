"""
Inference Interface Component for Streamlit Inference Module
===========================================================

Handles the main inference interface with chat, streaming, and result display.
"""

import streamlit as st
import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class InferenceInterfaceComponent:
    """Component for handling inference operations."""
    
    def __init__(self):
        """Initialize inference interface."""
        pass
    
    def render(self, inference_engine, available_agents: Dict[str, Any]):
        """Render inference interface."""
        st.markdown("## 🧠 Inference Interface")
        
        if not available_agents:
            st.warning("No agents available. Please create an agent first in the Agent Management section.")
            return
        
        # Main interface tabs
        tab1, tab2, tab3 = st.tabs(["💬 Chat Interface", "⚙️ Batch Processing", "📊 Results History"])
        
        with tab1:
            self._render_chat_interface(inference_engine, available_agents)
        
        with tab2:
            self._render_batch_interface(inference_engine, available_agents)
        
        with tab3:
            self._render_results_history()
    
    def _render_chat_interface(self, inference_engine, available_agents: Dict[str, Any]):
        """Render real-time chat interface."""
        st.markdown("### 💬 Real-time Chat")
        
        # Agent selection and settings
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Agent selection
            agent_options = {f"{data['name']} ({data['model']['provider']})": agent_id 
                           for agent_id, data in available_agents.items() if data.get('active', True)}
            
            if not agent_options:
                st.warning("No active agents available.")
                return
            
            selected_agent_display = st.selectbox("Select Agent", list(agent_options.keys()))
            selected_agent_id = agent_options[selected_agent_display]
            selected_agent = available_agents[selected_agent_id]
        
        with col2:
            # Quick settings
            with st.expander("⚙️ Quick Settings"):
                stream_enabled = st.checkbox("Enable Streaming", value=True)
                show_reasoning = st.checkbox("Show Reasoning", value=False)
                show_tools = st.checkbox("Show Tool Calls", value=True)
                temperature = st.slider("Temperature", 0.0, 2.0, 
                                       selected_agent.get('model', {}).get('temperature', 0.7), 0.1)
        
        # Agent info display
        st.markdown(f"**Selected Agent:** {selected_agent['name']}")
        st.caption(f"Model: {selected_agent['model']['provider']} - {selected_agent['model']['model_name']}")
        st.caption(f"Description: {selected_agent.get('description', 'No description')}")
        
        # Chat interface
        self._render_chat_messages()
        
        # Input area
        with st.container():
            query_input = st.text_area(
                "Enter your message:",
                height=100,
                placeholder="Type your message here...",
                key="chat_input"
            )
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                send_clicked = st.button("🚀 Send Message", use_container_width=True)
            
            with col2:
                clear_clicked = st.button("🗑️ Clear Chat", use_container_width=True)
            
            with col3:
                save_clicked = st.button("💾 Save Chat", use_container_width=True)
            
            with col4:
                export_clicked = st.button("📤 Export", use_container_width=True)
        
        # Handle button clicks
        if send_clicked and query_input.strip():
            self._handle_chat_message(
                inference_engine, 
                selected_agent_id, 
                selected_agent,
                query_input,
                {
                    "stream": stream_enabled,
                    "show_tool_calls": show_tools,
                    "show_full_reasoning": show_reasoning,
                    "temperature": temperature
                }
            )
        
        if clear_clicked:
            st.session_state.chat_messages = []
            st.rerun()
        
        if save_clicked:
            self._save_chat_session()
        
        if export_clicked:
            self._export_chat_session()
    
    def _render_chat_messages(self):
        """Render chat message history."""
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_messages):
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                        st.caption(f"📅 {message.get('timestamp', 'Unknown time')}")
                
                elif message["role"] == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])
                        
                        # Show additional info if available
                        if message.get("tool_calls"):
                            with st.expander("🔧 Tool Calls"):
                                for tool_call in message["tool_calls"]:
                                    st.json(tool_call)
                        
                        if message.get("reasoning"):
                            with st.expander("🧠 Reasoning"):
                                st.markdown(message["reasoning"])
                        
                        # Metrics
                        if message.get("metrics"):
                            metrics = message["metrics"]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Duration", f"{metrics.get('duration_seconds', 0):.2f}s")
                            with col2:
                                st.metric("Tokens", metrics.get('tokens_used', 'N/A'))
                            with col3:
                                st.metric("Tools", metrics.get('tool_calls_count', 0))
                        
                        st.caption(f"📅 {message.get('timestamp', 'Unknown time')} | "
                                 f"🤖 {message.get('agent_name', 'Unknown agent')}")
                
                elif message["role"] == "system":
                    with st.chat_message("system"):
                        st.info(message["content"])
    
    def _handle_chat_message(self, inference_engine, agent_id: str, agent_data: Dict[str, Any], 
                           query: str, settings: Dict[str, Any]):
        """Handle sending a chat message."""
        try:
            # Add user message to chat
            user_message = {
                "role": "user",
                "content": query,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "agent_id": agent_id
            }
            st.session_state.chat_messages.append(user_message)
            
            # Show typing indicator
            with st.chat_message("assistant"):
                typing_placeholder = st.empty()
                typing_placeholder.markdown("🤔 Thinking...")
                
                # Execute inference
                try:
                    # Create a mock agent instance for the inference engine
                    mock_agent = self._create_mock_agent(agent_data)
                    
                    # Run inference
                    if settings.get("stream", True):
                        result = self._run_streaming_inference(
                            inference_engine, mock_agent, query, settings, typing_placeholder
                        )
                    else:
                        result = self._run_standard_inference(
                            inference_engine, mock_agent, query, settings
                        )
                    
                    if result and result.get("success"):
                        # Add assistant message to chat
                        assistant_message = {
                            "role": "assistant",
                            "content": result["result"]["response_content"],
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "agent_id": agent_id,
                            "agent_name": agent_data["name"],
                            "tool_calls": result["result"].get("tool_calls", []),
                            "reasoning": result["result"].get("reasoning_content"),
                            "metrics": result["result"].get("metrics")
                        }
                        st.session_state.chat_messages.append(assistant_message)
                        
                        # Add to inference history
                        st.session_state.inference_history.append(result["result"])
                        
                        typing_placeholder.empty()
                        st.rerun()
                    
                    else:
                        typing_placeholder.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    typing_placeholder.error(f"❌ Inference failed: {str(e)}")
                    logger.error(f"Chat inference error: {e}", exc_info=True)
        
        except Exception as e:
            st.error(f"Failed to process message: {str(e)}")
            logger.error(f"Chat message processing error: {e}", exc_info=True)
    
    def _create_mock_agent(self, agent_data: Dict[str, Any]):
        """Create a mock agent instance for inference."""
        class MockAgent:
            def __init__(self, agent_data):
                self.name = agent_data["name"]
                self.id = agent_data["id"]
                self.model_info = agent_data["model"]
                self.system_prompt = agent_data.get("system_prompt", "")
                self.tools = agent_data.get("tools", [])
                self.knowledge_bases = agent_data.get("knowledge_bases", [])
            
            def run(self, query: str, **kwargs):
                # This is a simplified mock implementation
                # In a real implementation, this would use the actual AGNO framework
                response_content = f"Mock response from {self.name} for query: {query}"
                
                class MockResponse:
                    def __init__(self):
                        self.content = response_content
                        self.reasoning_content = f"Reasoning for: {query}" if kwargs.get("show_full_reasoning") else None
                
                return MockResponse()
        
        return MockAgent(agent_data)
    
    def _run_streaming_inference(self, inference_engine, agent, query: str, 
                               settings: Dict[str, Any], placeholder):
        """Run streaming inference (simplified)."""
        try:
            # Simulate streaming response
            placeholder.markdown("🤔 Processing your request...")
            time.sleep(1)
            
            # Run actual inference (simplified)
            response = agent.run(query, **settings)
            
            # Simulate metrics
            metrics = {
                "duration_seconds": 1.5,
                "response_size_chars": len(response.content),
                "tool_calls_count": 0,
                "reasoning_steps": 1 if settings.get("show_full_reasoning") else 0
            }
            
            return {
                "success": True,
                "result": {
                    "response_content": response.content,
                    "reasoning_content": response.reasoning_content,
                    "tool_calls": [],
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _run_standard_inference(self, inference_engine, agent, query: str, settings: Dict[str, Any]):
        """Run standard (non-streaming) inference."""
        return self._run_streaming_inference(inference_engine, agent, query, settings, None)
    
    def _render_batch_interface(self, inference_engine, available_agents: Dict[str, Any]):
        """Render batch processing interface."""
        st.markdown("### ⚙️ Batch Processing")
        
        # Batch configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📝 Batch Queries")
            
            # Input methods
            input_method = st.radio("Input Method", ["Manual Entry", "File Upload", "Template"])
            
            queries = []
            
            if input_method == "Manual Entry":
                query_text = st.text_area(
                    "Enter queries (one per line):",
                    height=200,
                    placeholder="Query 1\nQuery 2\nQuery 3..."
                )
                if query_text:
                    queries = [q.strip() for q in query_text.split('\n') if q.strip()]
            
            elif input_method == "File Upload":
                uploaded_file = st.file_uploader("Upload query file", type=['txt', 'csv'])
                if uploaded_file:
                    content = uploaded_file.read().decode('utf-8')
                    if uploaded_file.name.endswith('.csv'):
                        # Handle CSV format
                        import csv
                        import io
                        reader = csv.reader(io.StringIO(content))
                        queries = [row[0] for row in reader if row]
                    else:
                        queries = [q.strip() for q in content.split('\n') if q.strip()]
            
            elif input_method == "Template":
                template_type = st.selectbox("Select Template", [
                    "Product Analysis",
                    "Customer Feedback",
                    "Content Generation",
                    "Data Analysis"
                ])
                
                templates = self._get_batch_templates()
                if template_type in templates:
                    queries = templates[template_type]
                    st.info(f"Loaded {len(queries)} template queries")
        
        with col2:
            st.markdown("#### ⚙️ Batch Settings")
            
            # Agent selection for batch
            agent_options = {f"{data['name']}": agent_id 
                           for agent_id, data in available_agents.items() if data.get('active', True)}
            
            batch_agent = st.selectbox("Select Agent for Batch", list(agent_options.keys()))
            
            # Batch processing options
            parallel_processing = st.checkbox("Parallel Processing", value=False)
            max_concurrent = st.slider("Max Concurrent", 1, 5, 2) if parallel_processing else 1
            
            delay_between = st.slider("Delay Between Queries (seconds)", 0.0, 5.0, 1.0, 0.5)
            
            # Output options
            save_results = st.checkbox("Save Results", value=True)
            export_format = st.selectbox("Export Format", ["JSON", "CSV", "Markdown"])
        
        # Preview and execute
        if queries:
            st.markdown(f"#### 📋 Query Preview ({len(queries)} queries)")
            
            with st.expander("View Queries"):
                for i, query in enumerate(queries[:10], 1):
                    st.markdown(f"**{i}.** {query}")
                if len(queries) > 10:
                    st.caption(f"... and {len(queries) - 10} more queries")
            
            # Execute batch
            if st.button(f"🚀 Execute Batch ({len(queries)} queries)", use_container_width=True):
                self._execute_batch_processing(
                    inference_engine,
                    available_agents[agent_options[batch_agent]],
                    queries,
                    {
                        "parallel": parallel_processing,
                        "max_concurrent": max_concurrent,
                        "delay": delay_between,
                        "save_results": save_results,
                        "export_format": export_format
                    }
                )
        else:
            st.info("Enter or upload queries to begin batch processing.")
    
    def _get_batch_templates(self) -> Dict[str, List[str]]:
        """Get predefined batch query templates."""
        return {
            "Product Analysis": [
                "Analyze the features of this product",
                "What are the main benefits?",
                "Who is the target audience?",
                "What are potential drawbacks?",
                "How does it compare to competitors?"
            ],
            "Customer Feedback": [
                "Summarize this customer review",
                "What is the sentiment?",
                "What are the main issues mentioned?",
                "What improvements are suggested?",
                "Rate the overall satisfaction"
            ],
            "Content Generation": [
                "Create a blog post introduction",
                "Generate social media content",
                "Write product descriptions",
                "Create email subject lines",
                "Generate FAQ responses"
            ],
            "Data Analysis": [
                "Analyze this dataset",
                "What are the key insights?",
                "Identify trends and patterns",
                "What are the outliers?",
                "Provide recommendations"
            ]
        }
    
    def _execute_batch_processing(self, inference_engine, agent_data: Dict[str, Any], 
                                queries: List[str], settings: Dict[str, Any]):
        """Execute batch processing of queries."""
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            results = []
            
            for i, query in enumerate(queries):
                # Update progress
                progress = (i + 1) / len(queries)
                progress_bar.progress(progress)
                status_text.text(f"Processing query {i + 1} of {len(queries)}: {query[:50]}...")
                
                # Process query
                try:
                    mock_agent = self._create_mock_agent(agent_data)
                    result = self._run_standard_inference(inference_engine, mock_agent, query, {})
                    
                    if result and result.get("success"):
                        results.append({
                            "query": query,
                            "response": result["result"]["response_content"],
                            "success": True,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        results.append({
                            "query": query,
                            "error": result.get("error", "Unknown error"),
                            "success": False,
                            "timestamp": datetime.now().isoformat()
                        })
                
                except Exception as e:
                    results.append({
                        "query": query,
                        "error": str(e),
                        "success": False,
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Delay between queries
                if settings.get("delay", 0) > 0 and i < len(queries) - 1:
                    time.sleep(settings["delay"])
            
            # Show results
            progress_bar.progress(1.0)
            status_text.text("✅ Batch processing completed!")
            
            successful = sum(1 for r in results if r["success"])
            failed = len(results) - successful
            
            st.success(f"Batch processing completed! ✅ {successful} successful, ❌ {failed} failed")
            
            # Display results summary
            with results_container:
                self._display_batch_results(results, settings.get("export_format", "JSON"))
        
        except Exception as e:
            st.error(f"Batch processing failed: {str(e)}")
            logger.error(f"Batch processing error: {e}", exc_info=True)
    
    def _display_batch_results(self, results: List[Dict[str, Any]], export_format: str):
        """Display batch processing results."""
        st.markdown("#### 📊 Batch Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        
        with col1:
            st.metric("Total Queries", len(results))
        with col2:
            st.metric("Successful", successful)
        with col3:
            st.metric("Failed", failed)
        with col4:
            success_rate = (successful / len(results)) * 100 if results else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Results display
        tab1, tab2, tab3 = st.tabs(["📋 Results List", "✅ Successful", "❌ Failed"])
        
        with tab1:
            for i, result in enumerate(results, 1):
                with st.expander(f"Query {i}: {result['query'][:50]}..."):
                    if result["success"]:
                        st.success("✅ Success")
                        st.markdown("**Response:**")
                        st.markdown(result["response"])
                    else:
                        st.error("❌ Failed")
                        st.markdown("**Error:**")
                        st.error(result["error"])
                    
                    st.caption(f"Timestamp: {result['timestamp']}")
        
        with tab2:
            successful_results = [r for r in results if r["success"]]
            if successful_results:
                for i, result in enumerate(successful_results, 1):
                    st.markdown(f"**{i}. {result['query']}**")
                    st.markdown(result["response"])
                    st.divider()
            else:
                st.info("No successful results")
        
        with tab3:
            failed_results = [r for r in results if not r["success"]]
            if failed_results:
                for i, result in enumerate(failed_results, 1):
                    st.markdown(f"**{i}. {result['query']}**")
                    st.error(result["error"])
                    st.divider()
            else:
                st.success("No failed results")
        
        # Export results
        if st.button("📤 Export Results", use_container_width=True):
            self._export_batch_results(results, export_format)
    
    def _export_batch_results(self, results: List[Dict[str, Any]], format: str):
        """Export batch results in specified format."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format == "JSON":
                content = json.dumps(results, indent=2, ensure_ascii=False)
                st.download_button(
                    "📥 Download JSON",
                    data=content,
                    file_name=f"batch_results_{timestamp}.json",
                    mime="application/json"
                )
            
            elif format == "CSV":
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Header
                writer.writerow(["Query", "Response", "Success", "Error", "Timestamp"])
                
                # Data
                for result in results:
                    writer.writerow([
                        result["query"],
                        result.get("response", ""),
                        result["success"],
                        result.get("error", ""),
                        result["timestamp"]
                    ])
                
                st.download_button(
                    "📥 Download CSV",
                    data=output.getvalue(),
                    file_name=f"batch_results_{timestamp}.csv",
                    mime="text/csv"
                )
            
            elif format == "Markdown":
                content = f"# Batch Processing Results\n\n"
                content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                for i, result in enumerate(results, 1):
                    content += f"## Query {i}\n\n"
                    content += f"**Query:** {result['query']}\n\n"
                    
                    if result["success"]:
                        content += f"**Status:** ✅ Success\n\n"
                        content += f"**Response:** {result['response']}\n\n"
                    else:
                        content += f"**Status:** ❌ Failed\n\n"
                        content += f"**Error:** {result['error']}\n\n"
                    
                    content += f"**Timestamp:** {result['timestamp']}\n\n"
                    content += "---\n\n"
                
                st.download_button(
                    "📥 Download Markdown",
                    data=content,
                    file_name=f"batch_results_{timestamp}.md",
                    mime="text/markdown"
                )
        
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
    
    def _render_results_history(self):
        """Render inference results history."""
        st.markdown("### 📊 Results History")
        
        if not st.session_state.inference_history:
            st.info("No inference history available.")
            return
        
        # History filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_last = st.selectbox("Show Last", [10, 25, 50, 100, "All"])
        
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.inference_history = []
                st.rerun()
        
        with col3:
            if st.button("📤 Export History"):
                self._export_inference_history()
        
        # Display history
        history = st.session_state.inference_history
        if show_last != "All":
            history = history[-show_last:]
        
        for i, result in enumerate(reversed(history), 1):
            with st.expander(f"Result {i}: {result.get('request', {}).get('query', 'Unknown')[:50]}..."):
                
                # Basic info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Query:**")
                    st.markdown(result.get('request', {}).get('query', 'Unknown'))
                
                with col2:
                    st.markdown("**Response:**")
                    st.markdown(result.get('response_content', 'No response'))
                
                # Metrics
                metrics = result.get('metrics', {})
                if metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Duration", f"{metrics.get('duration_seconds', 0):.2f}s")
                    with col2:
                        st.metric("Characters", metrics.get('response_size_chars', 0))
                    with col3:
                        st.metric("Tool Calls", metrics.get('tool_calls_count', 0))
                    with col4:
                        st.metric("Reasoning", metrics.get('reasoning_steps', 0))
                
                # Additional details
                if result.get('tool_calls'):
                    st.markdown("**Tool Calls:**")
                    st.json(result['tool_calls'])
                
                if result.get('reasoning_content'):
                    st.markdown("**Reasoning:**")
                    st.markdown(result['reasoning_content'])
                
                st.caption(f"Timestamp: {result.get('timestamp', 'Unknown')}")
    
    def _export_inference_history(self):
        """Export inference history."""
        try:
            content = json.dumps(st.session_state.inference_history, indent=2, ensure_ascii=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                "📥 Download History",
                data=content,
                file_name=f"inference_history_{timestamp}.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
    
    def _save_chat_session(self):
        """Save current chat session."""
        if st.session_state.chat_messages:
            st.success("💾 Chat session saved! (Feature in development)")
        else:
            st.warning("No chat messages to save.")
    
    def _export_chat_session(self):
        """Export current chat session."""
        try:
            if not st.session_state.chat_messages:
                st.warning("No chat messages to export.")
                return
            
            # Create formatted export
            content = f"# Chat Session Export\n\n"
            content += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            for message in st.session_state.chat_messages:
                if message["role"] == "user":
                    content += f"**User ({message.get('timestamp', 'Unknown')}):**\n"
                    content += f"{message['content']}\n\n"
                
                elif message["role"] == "assistant":
                    content += f"**Assistant ({message.get('timestamp', 'Unknown')}):**\n"
                    content += f"{message['content']}\n\n"
                    
                    if message.get("reasoning"):
                        content += f"*Reasoning:* {message['reasoning']}\n\n"
                
                content += "---\n\n"
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                "📥 Download Chat",
                data=content,
                file_name=f"chat_session_{timestamp}.md",
                mime="text/markdown"
            )
        
        except Exception as e:
            st.error(f"Export failed: {str(e)}")