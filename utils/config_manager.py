"""
Configuration Manager for Streamlit Inference Module
====================================================

Handles loading, saving, and managing application configuration.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: Path):
        """Initialize configuration manager."""
        self.config_path = config_path
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.default_config = {
            "app": {
                "name": "AI Agent Inference Module",
                "version": "1.0.0",
                "max_users": 5,
                "created": datetime.now().isoformat()
            },
            "models": {
                "ollama": {
                    "enabled": True,
                    "base_url": "http://localhost:11434",
                    "selected_model": None,
                    "available_models": []
                },
                "openai": {
                    "enabled": False,
                    "api_key": "",
                    "model": "gpt-4o",
                    "base_url": "https://api.openai.com/v1"
                },
                "anthropic": {
                    "enabled": False,
                    "api_key": "",
                    "model": "claude-3-sonnet-20240229"
                },
                "gemini": {
                    "enabled": False,
                    "api_key": "",
                    "model": "gemini-pro"
                }
            },
            "inference": {
                "default_stream": True,
                "default_temperature": 0.7,
                "default_max_tokens": 2048,
                "show_tool_calls": True,
                "show_reasoning": False,
                "cache_responses": True,
                "save_history": True
            },
            "memory": {
                "enabled": True,
                "max_memories_per_user": 1000,
                "auto_summarize": True,
                "retention_days": 30
            },
            "knowledge": {
                "enabled": True,
                "max_documents": 100,
                "chunk_size": 1000,
                "overlap": 200
            },
            "ui": {
                "theme": "professional",
                "sidebar_expanded": True,
                "show_advanced_options": False,
                "auto_refresh": True
            },
            "system_prompts": {
                "default": "You are a helpful AI assistant. Provide clear, accurate, and helpful responses.",
                "creative": "You are a creative AI assistant. Think outside the box and provide innovative solutions.",
                "analytical": "You are an analytical AI assistant. Focus on data-driven insights and logical reasoning.",
                "custom_prompts": {}
            }
        }
        
        # Load existing config or create default
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Merge with defaults to ensure all keys exist
                merged_config = self._merge_configs(self.default_config, config)
                logger.info(f"Configuration loaded from {self.config_path}")
                return merged_config
            else:
                # Create default config
                self.save_config(self.default_config)
                logger.info(f"Default configuration created at {self.config_path}")
                return self.default_config.copy()
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
            return self.default_config.copy()
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Save configuration to file."""
        try:
            config_to_save = config or self.config
            config_to_save["app"]["last_updated"] = datetime.now().isoformat()
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=2, ensure_ascii=False)
            
            if config:
                self.config = config
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        try:
            keys = key_path.split('.')
            value = self.config
            
            for key in keys:
                value = value[key]
            
            return value
            
        except (KeyError, TypeError):
            logger.warning(f"Configuration key not found: {key_path}")
            return default
    
    def set(self, key_path: str, value: Any) -> bool:
        """Set configuration value using dot notation."""
        try:
            keys = key_path.split('.')
            config = self.config
            
            # Navigate to parent of target key
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            # Set the value
            config[keys[-1]] = value
            
            # Save updated config
            return self.save_config()
            
        except Exception as e:
            logger.error(f"Error setting configuration value: {e}")
            return False
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> bool:
        """Update entire configuration section."""
        try:
            if section in self.config:
                self.config[section].update(updates)
            else:
                self.config[section] = updates
            
            return self.save_config()
            
        except Exception as e:
            logger.error(f"Error updating configuration section: {e}")
            return False
    
    def get_model_config(self, provider: str) -> Dict[str, Any]:
        """Get model configuration for specific provider."""
        return self.get(f"models.{provider}", {})
    
    def set_model_config(self, provider: str, config: Dict[str, Any]) -> bool:
        """Set model configuration for specific provider."""
        return self.update_section(f"models.{provider}", config)
    
    def get_system_prompt(self, prompt_name: str = "default") -> str:
        """Get system prompt by name."""
        # Check custom prompts first
        custom_prompts = self.get("system_prompts.custom_prompts", {})
        if prompt_name in custom_prompts:
            return custom_prompts[prompt_name]
        
        # Fall back to default prompts
        return self.get(f"system_prompts.{prompt_name}", 
                       self.get("system_prompts.default", "You are a helpful AI assistant."))
    
    def save_custom_prompt(self, name: str, prompt: str) -> bool:
        """Save a custom system prompt."""
        custom_prompts = self.get("system_prompts.custom_prompts", {})
        custom_prompts[name] = prompt
        return self.set("system_prompts.custom_prompts", custom_prompts)
    
    def delete_custom_prompt(self, name: str) -> bool:
        """Delete a custom system prompt."""
        custom_prompts = self.get("system_prompts.custom_prompts", {})
        if name in custom_prompts:
            del custom_prompts[name]
            return self.set("system_prompts.custom_prompts", custom_prompts)
        return False
    
    def get_all_prompts(self) -> Dict[str, str]:
        """Get all available system prompts."""
        prompts = {}
        
        # Add default prompts
        for key in ["default", "creative", "analytical"]:
            prompts[key] = self.get(f"system_prompts.{key}", "")
        
        # Add custom prompts
        custom_prompts = self.get("system_prompts.custom_prompts", {})
        prompts.update(custom_prompts)
        
        return prompts
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge user config with default config."""
        merged = default.copy()
        
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults."""
        try:
            self.config = self.default_config.copy()
            return self.save_config()
        except Exception as e:
            logger.error(f"Error resetting configuration: {e}")
            return False
    
    def export_config(self, export_path: Path) -> bool:
        """Export configuration to file."""
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, import_path: Path) -> bool:
        """Import configuration from file."""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
            
            # Merge with defaults to ensure integrity
            merged_config = self._merge_configs(self.default_config, imported_config)
            
            return self.save_config(merged_config)
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False