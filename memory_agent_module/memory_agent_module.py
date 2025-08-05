"""
MODULAR AI PIPELINE - Memory Agent Module
===============================================

Classe: MemoryAgentModule
Funzione: Gestisce agenti con memoria persistente per conversazioni contestualizzate
Metodo principale: execute()

Caratteristiche:
- Memoria utente persistente
- Riassunti sessione automatici
- Storage conversazioni SQLite
- Gestione multi-utente
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from dataclasses import dataclass, asdict

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.storage.sqlite import SqliteStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.reasoning import ReasoningTools


@dataclass
class MemoryConfig:
    """Configurazione per agente con memoria"""
    user_id: str
    agent_name: str
    enable_user_memories: bool = True
    enable_session_summaries: bool = True
    memory_db_file: str = "tmp/agent_memory.db"
    storage_db_file: str = "tmp/agent_storage.db"
    num_history_runs: int = 5
    model_type: str = "openai"
    model_id: str = "gpt-4o"
    tools: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryAgentModule:
    """Modulo per gestione agenti con memoria persistente"""
    
    def __init__(self, pipeline_controller: Any):
        """Inizializza il modulo con il controller della pipeline"""
        self.pipeline_controller = pipeline_controller
        self.agents_cache: Dict[str, Agent] = {}
        logging.info("MemoryAgentModule inizializzato")
    
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Punto di ingresso principale per eseguire azioni.
        
        Actions:
        - create_memory_agent: Crea agente con memoria
        - chat: Interagisce con agente esistente
        - get_memories: Recupera memorie utente
        - get_session_summary: Recupera riassunto sessione
        - clear_memories: Pulisce memorie utente
        """
        try:
            if action == "create_memory_agent":
                return await self._create_memory_agent(**kwargs)
            elif action == "chat":
                return await self._chat_with_agent(**kwargs)
            elif action == "get_memories":
                return await self._get_user_memories(**kwargs)
            elif action == "get_session_summary":
                return await self._get_session_summary(**kwargs)
            elif action == "clear_memories":
                return await self._clear_memories(**kwargs)
            else:
                raise ValueError(f"Azione non supportata: {action}")
                
        except Exception as e:
            logging.error(f"Errore in MemoryAgentModule.execute: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _create_memory_agent(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crea un agente con memoria persistente"""
        try:
            # Parse configurazione
            memory_config = MemoryConfig(**config)
            
            # Crea sistema memoria
            memory = Memory(
                model=self._get_model(memory_config.model_type, memory_config.model_id),
                db=SqliteMemoryDb(
                    table_name=f"memories_{memory_config.agent_name}",
                    db_file=memory_config.memory_db_file
                ),
                delete_memories=True,
                clear_memories=True
            )
            
            # Crea storage conversazioni
            storage = SqliteStorage(
                table_name=f"sessions_{memory_config.agent_name}",
                db_file=memory_config.storage_db_file
            )
            
            # Configura tools
            tools = []
            if memory_config.tools:
                for tool_name in memory_config.tools:
                    if tool_name == "duckduckgo":
                        tools.append(DuckDuckGoTools())
                    elif tool_name == "reasoning":
                        tools.append(ReasoningTools(add_instructions=True))
            
            # Crea agente
            agent = Agent(
                name=memory_config.agent_name,
                model=self._get_model(memory_config.model_type, memory_config.model_id),
                user_id=memory_config.user_id,
                memory=memory,
                storage=storage,
                tools=tools,
                enable_agentic_memory=True,
                enable_user_memories=memory_config.enable_user_memories,
                enable_session_summaries=memory_config.enable_session_summaries,
                add_history_to_messages=True,
                num_history_responses=memory_config.num_history_runs,
                instructions=[
                    "Sei un assistente intelligente con memoria persistente.",
                    "Ricorda i dettagli importanti sugli utenti e fai riferimento ad essi naturalmente.",
                    "Mantieni un tono cordiale e positivo.",
                    "Quando appropriato, fai riferimento a conversazioni precedenti."
                ],
                markdown=True
            )
            
            # Cache agente
            agent_key = f"{memory_config.user_id}_{memory_config.agent_name}"
            self.agents_cache[agent_key] = agent
            
            logging.info(f"Agente con memoria creato: {agent_key}")
            
            return {
                "success": True,
                "agent_key": agent_key,
                "config": memory_config.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Errore creazione agente con memoria: {e}")
            raise
    
    async def _chat_with_agent(
        self,
        user_id: str,
        agent_name: str,
        message: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Interagisce con un agente esistente"""
        try:
            agent_key = f"{user_id}_{agent_name}"
            
            # Recupera agente dalla cache
            if agent_key not in self.agents_cache:
                return {
                    "success": False,
                    "error": f"Agente non trovato: {agent_key}",
                    "timestamp": datetime.now().isoformat()
                }
            
            agent = self.agents_cache[agent_key]
            
            # Imposta session_id se fornito
            if session_id:
                agent.session_id = session_id
            
            # Esegui conversazione
            response = agent.run(message)
            
            # Estrai informazioni memoria
            memory_info = {
                "session_id": agent.session_id,
                "memories_count": len(agent.memory.get_user_memories(user_id))
                if hasattr(agent, 'memory') and agent.memory else 0
            }
            
            return {
                "success": True,
                "response": response.content,
                "memory_info": memory_info,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Errore chat con agente: {e}")
            raise
    
    async def _get_user_memories(
        self,
        user_id: str,
        agent_name: str
    ) -> Dict[str, Any]:
        """Recupera le memorie di un utente"""
        try:
            agent_key = f"{user_id}_{agent_name}"
            
            if agent_key not in self.agents_cache:
                return {
                    "success": False,
                    "error": f"Agente non trovato: {agent_key}",
                    "timestamp": datetime.now().isoformat()
                }
            
            agent = self.agents_cache[agent_key]
            
            # Recupera memorie
            memories = []
            if hasattr(agent, 'memory') and agent.memory:
                user_memories = agent.memory.get_user_memories(user_id)
                memories = [memory.to_dict() for memory in user_memories]
            
            return {
                "success": True,
                "user_id": user_id,
                "memories": memories,
                "count": len(memories),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Errore recupero memorie: {e}")
            raise
    
    async def _get_session_summary(
        self,
        user_id: str,
        agent_name: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Recupera il riassunto di una sessione"""
        try:
            agent_key = f"{user_id}_{agent_name}"
            
            if agent_key not in self.agents_cache:
                return {
                    "success": False,
                    "error": f"Agente non trovato: {agent_key}",
                    "timestamp": datetime.now().isoformat()
                }
            
            agent = self.agents_cache[agent_key]
            
            # Recupera riassunti
            summaries = []
            if hasattr(agent, 'memory') and agent.memory:
                session_summaries = agent.memory.get_session_summaries(user_id)
                if session_id:
                    summaries = [
                        s.to_dict() for s in session_summaries 
                        if s.session_id == session_id
                    ]
                else:
                    summaries = [s.to_dict() for s in session_summaries]
            
            return {
                "success": True,
                "user_id": user_id,
                "session_id": session_id,
                "summaries": summaries,
                "count": len(summaries),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Errore recupero riassunto sessione: {e}")
            raise
    
    async def _clear_memories(
        self,
        user_id: str,
        agent_name: str
    ) -> Dict[str, Any]:
        """Pulisce le memorie di un utente"""
        try:
            agent_key = f"{user_id}_{agent_name}"
            
            if agent_key not in self.agents_cache:
                return {
                    "success": False,
                    "error": f"Agente non trovato: {agent_key}",
                    "timestamp": datetime.now().isoformat()
                }
            
            agent = self.agents_cache[agent_key]
            
            # Pulisci memorie
            if hasattr(agent, 'memory') and agent.memory:
                agent.memory.clear_user_memories(user_id)
            
            return {
                "success": True,
                "user_id": user_id,
                "message": "Memorie utente cancellate",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Errore pulizia memorie: {e}")
            raise
    
    def _get_model(self, model_type: str, model_id: str):
        """Factory per creazione modelli"""
        if model_type == "openai":
            return OpenAIChat(id=model_id)
        elif model_type == "anthropic":
            return Claude(id=model_id)
        else:
            raise ValueError(f"Tipo modello non supportato: {model_type}")


# Test standalone
if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.INFO)
    
    class MockPipelineController:
        async def execute_module(self, module: str, pid: str, **kwargs):
            print(f"MOCK: Chiamata a {module} con {kwargs}")
            return {"success": True, "result": "mock"}
    
    async def test_memory_agent():
        controller = MockPipelineController()
        module = MemoryAgentModule(controller)
        
        # Test creazione agente
        print("\n1. Creazione agente con memoria...")
        config = {
            "user_id": "test_user",
            "agent_name": "assistant",
            "tools": ["duckduckgo", "reasoning"],
            "enable_user_memories": True,
            "enable_session_summaries": True
        }
        result = await module.execute(action="create_memory_agent", config=config)
        print(f"Risultato: {result}")
        assert result["success"]
        
        # Test chat
        print("\n2. Test conversazione...")
        chat_result = await module.execute(
            action="chat",
            user_id="test_user",
            agent_name="assistant",
            message="Mi chiamo Mario e lavoro come sviluppatore"
        )
        print(f"Risposta: {chat_result}")
        
        # Test recupero memorie
        print("\n3. Recupero memorie...")
        memories = await module.execute(
            action="get_memories",
            user_id="test_user",
            agent_name="assistant"
        )
        print(f"Memorie: {memories}")
        
        print("\n✅ Test completato!")
    
    asyncio.run(test_memory_agent())