"""
Model Scanner for Streamlit Inference Module
============================================

Scans and manages available AI models from different providers.
"""

import requests
import logging
from typing import Dict, List, Optional, Any
import subprocess
import json

logger = logging.getLogger(__name__)


class ModelScanner:
    """Scans and manages available AI models."""
    
    def __init__(self):
        """Initialize model scanner."""
        self.ollama_base_url = "http://localhost:11434"
        self.cached_models = {}
        self.last_scan_time = None
    
    def scan_ollama_models(self, base_url: str = None) -> List[Dict[str, Any]]:
        """Scan available Ollama models."""
        try:
            url = base_url or self.ollama_base_url
            
            # Try to get models from Ollama API
            response = requests.get(f"{url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                models = []
                
                for model in data.get("models", []):
                    model_info = {
                        "name": model.get("name", "Unknown"),
                        "size": model.get("size", 0),
                        "digest": model.get("digest", ""),
                        "modified_at": model.get("modified_at", ""),
                        "details": model.get("details", {}),
                        "provider": "ollama",
                        "available": True
                    }
                    models.append(model_info)
                
                self.cached_models["ollama"] = models
                logger.info(f"Found {len(models)} Ollama models")
                return models
            else:
                logger.warning(f"Ollama API returned status {response.status_code}")
                return []
                
        except requests.exceptions.ConnectionError:
            logger.warning("Could not connect to Ollama - service may not be running")
            return []
        except requests.exceptions.Timeout:
            logger.warning("Ollama API request timed out")
            return []
        except Exception as e:
            logger.error(f"Error scanning Ollama models: {e}")
            return []
    
    def check_ollama_status(self, base_url: str = None) -> Dict[str, Any]:
        """Check Ollama service status."""
        try:
            url = base_url or self.ollama_base_url
            
            # Try to ping Ollama
            response = requests.get(f"{url}/api/version", timeout=3)
            
            if response.status_code == 200:
                version_info = response.json()
                return {
                    "status": "online",
                    "version": version_info.get("version", "unknown"),
                    "url": url,
                    "message": "Ollama is running"
                }
            else:
                return {
                    "status": "error",
                    "version": None,
                    "url": url,
                    "message": f"Ollama returned status {response.status_code}"
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "status": "offline",
                "version": None,
                "url": url,
                "message": "Cannot connect to Ollama service"
            }
        except Exception as e:
            return {
                "status": "error",
                "version": None,
                "url": url,
                "message": f"Error checking Ollama: {str(e)}"
            }
    
    def install_ollama_model(self, model_name: str, base_url: str = None) -> Dict[str, Any]:
        """Install/pull an Ollama model."""
        try:
            url = base_url or self.ollama_base_url
            
            # Start model pull
            response = requests.post(
                f"{url}/api/pull",
                json={"name": model_name},
                timeout=300  # 5 minutes timeout for model download
            )
            
            if response.status_code == 200:
                logger.info(f"Model {model_name} installation started")
                return {
                    "success": True,
                    "message": f"Model {model_name} installation started",
                    "model_name": model_name
                }
            else:
                error_msg = f"Failed to install model {model_name}: {response.status_code}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "message": error_msg,
                    "model_name": model_name
                }
                
        except Exception as e:
            error_msg = f"Error installing model {model_name}: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "model_name": model_name
            }
    
    def remove_ollama_model(self, model_name: str, base_url: str = None) -> Dict[str, Any]:
        """Remove an Ollama model."""
        try:
            url = base_url or self.ollama_base_url
            
            response = requests.delete(
                f"{url}/api/delete",
                json={"name": model_name},
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Model {model_name} removed successfully")
                return {
                    "success": True,
                    "message": f"Model {model_name} removed successfully",
                    "model_name": model_name
                }
            else:
                error_msg = f"Failed to remove model {model_name}: {response.status_code}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "message": error_msg,
                    "model_name": model_name
                }
                
        except Exception as e:
            error_msg = f"Error removing model {model_name}: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "model_name": model_name
            }
    
    def get_popular_models(self) -> List[Dict[str, Any]]:
        """Get list of popular/recommended models for installation."""
        return [
            {
                "name": "llama3.1:8b",
                "description": "Meta's Llama 3.1 8B parameter model",
                "size": "4.7GB",
                "category": "general",
                "provider": "ollama",
                "recommended": True
            },
            {
                "name": "llama3.1:70b",
                "description": "Meta's Llama 3.1 70B parameter model (high quality)",
                "size": "40GB",
                "category": "general",
                "provider": "ollama",
                "recommended": False
            },
            {
                "name": "mistral:7b",
                "description": "Mistral 7B model - fast and efficient",
                "size": "4.1GB", 
                "category": "general",
                "provider": "ollama",
                "recommended": True
            },
            {
                "name": "codellama:7b",
                "description": "Code-specialized model based on Llama",
                "size": "3.8GB",
                "category": "coding",
                "provider": "ollama",
                "recommended": True
            },
            {
                "name": "phi3:mini",
                "description": "Microsoft Phi-3 Mini - compact and efficient",
                "size": "2.3GB",
                "category": "general",
                "provider": "ollama",
                "recommended": True
            },
            {
                "name": "gemma:7b",
                "description": "Google's Gemma 7B model",
                "size": "5.0GB",
                "category": "general",
                "provider": "ollama",
                "recommended": False
            },
            {
                "name": "qwen2:7b",
                "description": "Alibaba's Qwen2 7B model",
                "size": "4.4GB",
                "category": "general",
                "provider": "ollama",
                "recommended": False
            }
        ]
    
    def get_openai_models(self) -> List[Dict[str, Any]]:
        """Get available OpenAI models."""
        return [
            {
                "name": "gpt-4o",
                "description": "GPT-4 Omni - latest multimodal model",
                "provider": "openai",
                "category": "general",
                "multimodal": True,
                "context_length": 128000,
                "available": True
            },
            {
                "name": "gpt-4o-mini",
                "description": "GPT-4 Omni Mini - faster and cheaper",
                "provider": "openai",
                "category": "general",
                "multimodal": True,
                "context_length": 128000,
                "available": True
            },
            {
                "name": "gpt-4-turbo",
                "description": "GPT-4 Turbo - high capability model",
                "provider": "openai",
                "category": "general",
                "multimodal": False,
                "context_length": 128000,
                "available": True
            },
            {
                "name": "gpt-3.5-turbo",
                "description": "GPT-3.5 Turbo - fast and efficient",
                "provider": "openai",
                "category": "general",
                "multimodal": False,
                "context_length": 16385,
                "available": True
            }
        ]
    
    def get_anthropic_models(self) -> List[Dict[str, Any]]:
        """Get available Anthropic models."""
        return [
            {
                "name": "claude-3-5-sonnet-20240620",
                "description": "Claude 3.5 Sonnet - most capable model",
                "provider": "anthropic",
                "category": "general",
                "context_length": 200000,
                "available": True
            },
            {
                "name": "claude-3-opus-20240229",
                "description": "Claude 3 Opus - strongest performance",
                "provider": "anthropic",
                "category": "general",
                "context_length": 200000,
                "available": True
            },
            {
                "name": "claude-3-sonnet-20240229",
                "description": "Claude 3 Sonnet - balanced performance",
                "provider": "anthropic",
                "category": "general",
                "context_length": 200000,
                "available": True
            },
            {
                "name": "claude-3-haiku-20240307",
                "description": "Claude 3 Haiku - fastest model",
                "provider": "anthropic",
                "category": "general",
                "context_length": 200000,
                "available": True
            }
        ]
    
    def get_gemini_models(self) -> List[Dict[str, Any]]:
        """Get available Gemini models."""
        return [
            {
                "name": "gemini-1.5-pro",
                "description": "Gemini 1.5 Pro - most capable model",
                "provider": "gemini",
                "category": "general",
                "multimodal": True,
                "context_length": 2000000,
                "available": True
            },
            {
                "name": "gemini-1.5-flash",
                "description": "Gemini 1.5 Flash - faster and efficient",
                "provider": "gemini",
                "category": "general",
                "multimodal": True,
                "context_length": 1000000,
                "available": True
            },
            {
                "name": "gemini-pro",
                "description": "Gemini Pro - balanced performance",
                "provider": "gemini",
                "category": "general",
                "multimodal": False,
                "context_length": 32000,
                "available": True
            }
        ]
    
    def get_all_models(self, include_ollama: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """Get all available models organized by provider."""
        models = {
            "openai": self.get_openai_models(),
            "anthropic": self.get_anthropic_models(),
            "gemini": self.get_gemini_models()
        }
        
        if include_ollama:
            models["ollama"] = self.scan_ollama_models()
        
        return models
    
    def validate_model_config(self, provider: str, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model configuration."""
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": []
        }
        
        try:
            if provider == "ollama":
                # Check if Ollama is running
                status = self.check_ollama_status(config.get("base_url"))
                if status["status"] != "online":
                    validation_result["errors"].append(f"Ollama service not available: {status['message']}")
                    return validation_result
                
                # Check if model exists
                models = self.scan_ollama_models(config.get("base_url"))
                model_names = [m["name"] for m in models]
                if model_name not in model_names:
                    validation_result["errors"].append(f"Model '{model_name}' not found in Ollama")
                    return validation_result
            
            elif provider == "openai":
                if not config.get("api_key"):
                    validation_result["errors"].append("OpenAI API key is required")
                    return validation_result
            
            elif provider == "anthropic":
                if not config.get("api_key"):
                    validation_result["errors"].append("Anthropic API key is required")
                    return validation_result
            
            elif provider == "gemini":
                if not config.get("api_key"):
                    validation_result["errors"].append("Gemini API key is required")
                    return validation_result
            
            else:
                validation_result["errors"].append(f"Unknown provider: {provider}")
                return validation_result
            
            validation_result["valid"] = True
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def test_model_connection(self, provider: str, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test connection to a specific model."""
        test_result = {
            "success": False,
            "response_time": None,
            "error": None,
            "model_info": None
        }
        
        try:
            import time
            start_time = time.time()
            
            if provider == "ollama":
                url = config.get("base_url", self.ollama_base_url)
                
                # Test with a simple generation
                response = requests.post(
                    f"{url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": "Hello",
                        "stream": False,
                        "options": {"num_predict": 5}
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    test_result["success"] = True
                    test_result["model_info"] = {
                        "response": data.get("response", ""),
                        "model": data.get("model", model_name)
                    }
                else:
                    test_result["error"] = f"Ollama returned status {response.status_code}"
            
            # For other providers, we'd need their respective SDKs
            # This is a placeholder for now
            elif provider in ["openai", "anthropic", "gemini"]:
                test_result["success"] = True  # Placeholder
                test_result["model_info"] = {"note": "Connection test not implemented for this provider"}
            
            test_result["response_time"] = round(time.time() - start_time, 2)
            
        except Exception as e:
            test_result["error"] = str(e)
        
        return test_result