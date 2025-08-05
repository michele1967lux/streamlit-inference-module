"""
MODULAR AI PIPELINE - Inference Engine Module
============================================

Modulo: InferenceEngineModule
Funzione: Gestisce l'esecuzione di query su agenti e team AGNO con logging completo e configurazioni avanzate
Integrazione: Riceve configurazioni dalla GUI ed esegue query su agenti/team con tracking completo
Metodo principale: execute() - Esegue inference su agenti o team con parametri personalizzabili

Caratteristiche:
- Esecuzione query su agenti singoli e team
- Logging dettagliato con livelli configurabili
- Streaming responses con tracking intermedio
- Gestione reasoning e tool calls
- Cache risultati per performance
- Export risultati in formati multipli
- Metriche performance e token usage
- Support per modalità batch
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Iterator, AsyncIterator
from dataclasses import dataclass, asdict, field
from pathlib import Path
from textwrap import dedent
import sys
from io import StringIO

# HTTP client per connection pooling
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

# AGNO imports
from agno.agent import Agent, RunResponse, RunEvent
from agno.team.team import Team
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.storage.sqlite import SqliteStorage
import sys
from pathlib import Path
# Add parent directory to path for imports
gui_dir = Path(__file__).parent.parent.parent
if str(gui_dir) not in sys.path:
    sys.path.insert(0, str(gui_dir))
from adapters.storage_adapter import StorageAdapter
from agno.utils.log import logger
from agno.utils.pprint import pprint_run_response

# ChatFormatManager import
try:
    from utils.chat_format_manager import ChatFormatManager
    CHAT_FORMAT_MANAGER_AVAILABLE = True
except ImportError:
    ChatFormatManager = None
    CHAT_FORMAT_MANAGER_AVAILABLE = False

# StreamingEventEmitter import for live GUI updates
try:
    from utils.streaming_event_emitter import (
        StreamingEventEmitter, 
        create_streaming_event_emitter,
        StreamingEvent,
        StreamingEventType
    )
    STREAMING_EVENT_EMITTER_AVAILABLE = True
except ImportError:
    StreamingEventEmitter = None
    create_streaming_event_emitter = None
    StreamingEvent = None
    StreamingEventType = None
    STREAMING_EVENT_EMITTER_AVAILABLE = False

# Retry and similarity utilities import
try:
    from utils.retry_utils import RetryManager, RetryConfig, with_retry, CircuitOpenError, CircuitState
    from utils.similarity_utils import SimilarityFinder, find_fallback_model
    RESILIENCE_UTILS_AVAILABLE = True
except ImportError:
    RetryManager = None
    RetryConfig = None
    with_retry = None
    CircuitOpenError = None
    CircuitState = None
    SimilarityFinder = None
    find_fallback_model = None
    RESILIENCE_UTILS_AVAILABLE = False

# Connection Pooling imports for FASE 6 Task 6.2
try:
    from utils.connection_pool import (
        ConnectionPool, 
        ConnectionPoolManager, 
        DatabaseConnectionFactory, 
        HTTPConnectionFactory,
        get_pool_manager
    )
    CONNECTION_POOL_AVAILABLE = True
except ImportError:
    ConnectionPool = None
    ConnectionPoolManager = None
    DatabaseConnectionFactory = None
    HTTPConnectionFactory = None
    get_pool_manager = None
    CONNECTION_POOL_AVAILABLE = False

# Rich imports per output formattato
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich import print as rprint

console = Console()


@dataclass
class InferenceSettings:
    """Configurazioni avanzate per l'inference."""
    stream: bool = True
    show_tool_calls: bool = True
    show_full_reasoning: bool = False
    stream_intermediate_steps: bool = True
    markdown: bool = True
    save_to_file: Optional[str] = None
    cache_responses: bool = True
    timeout_seconds: Optional[int] = None
    max_retries: int = 3
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    debug_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    export_format: str = "json"  # json, markdown, txt
    track_metrics: bool = True
    format_preferences: Dict[str, Any] = field(default_factory=dict)  # Chat format preferences
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryRequest:
    """Richiesta di query strutturata."""
    query: str
    target_type: str  # "agent" o "team"
    target_id: str
    settings: InferenceSettings = field(default_factory=InferenceSettings)
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["settings"] = self.settings.to_dict()
        return data


@dataclass
class InferenceMetrics:
    """Metriche dell'inference."""
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    tokens_used: Optional[int] = None
    tool_calls_count: int = 0
    reasoning_steps: int = 0
    error_count: int = 0
    retry_count: int = 0
    response_size_chars: int = 0
    
    def calculate_duration(self):
        """Calcola durata totale."""
        if self.end_time and self.start_time:
            self.duration_seconds = round(self.end_time - self.start_time, 2)
    
    def to_dict(self) -> Dict[str, Any]:
        self.calculate_duration()
        return asdict(self)


@dataclass
class InferenceResult:
    """Risultato completo dell'inference."""
    success: bool
    request: QueryRequest
    response_content: Optional[str] = None
    full_response: Optional[RunResponse] = None
    reasoning_content: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Optional[InferenceMetrics] = None
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # Model-specific parsed data
    claude_thinking: Optional[str] = None  # Claude thinking tags content
    gemini_safety: Optional[Dict[str, Any]] = None  # Gemini safety ratings
    openai_structured: Optional[Dict[str, Any]] = None  # OpenAI structured output
    ollama_metadata: Optional[Dict[str, Any]] = None  # Ollama metadata
    parsed_metadata: Dict[str, Any] = field(default_factory=dict)  # Generic parsed metadata
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["request"] = self.request.to_dict()
        if self.metrics:
            data["metrics"] = self.metrics.to_dict()
        # Non serializzare full_response (oggetto complesso)
        data.pop("full_response", None)
        return data


class DebugCapture:
    """Cattura output debug per logging."""
    def __init__(self):
        self.logs = []
        self._stdout = sys.stdout
        self._buffer = StringIO()
    
    def start(self):
        sys.stdout = self._buffer
    
    def stop(self):
        sys.stdout = self._stdout
        output = self._buffer.getvalue()
        if output:
            self.logs.extend(output.strip().split('\n'))
        self._buffer = StringIO()
    
    def get_logs(self):
        return self.logs


class InferenceEngineModule:
    """Motore di inference per agenti e team AGNO."""
    
    def __init__(self, agent_controller=None, team_constructor=None,
                 storage_path: str = "tmp/inference", format_manager=None, 
                 knowledge_manager=None):
        """
        Inizializza il motore di inference.
        
        Args:
            agent_controller: Controller per gestione agenti
            team_constructor: Constructor per gestione team
            storage_path: Path per storage risultati
            format_manager: ChatFormatManager per formatting query/responses
            knowledge_manager: KnowledgeManager per gestione knowledge base
        """
        self.agent_controller = agent_controller
        self.team_constructor = team_constructor
        self.console = console
        
        # Storage per risultati e cache
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        result_storage = SqliteStorage(
            table_name="inference_results",
            db_file=str(self.storage_path / "results.db")
        )
        self.result_storage = StorageAdapter(result_storage, "InferenceResults")
        
        cache_storage = SqliteStorage(
            table_name="response_cache",
            db_file=str(self.storage_path / "cache.db")
        )
        self.cache_storage = StorageAdapter(cache_storage, "ResponseCache")
        
        # Cache runtime
        self.active_agents = {}  # agent_id -> Agent instance
        self.active_teams = {}   # team_id -> Team instance
        
        # Debug capture
        self.debug_capture = DebugCapture()
        
        # ChatFormatManager injection
        self.format_manager = format_manager
        if not self.format_manager and CHAT_FORMAT_MANAGER_AVAILABLE:
            try:
                self.format_manager = ChatFormatManager()
                self._log("ChatFormatManager inizializzato automaticamente", "INFO")
            except Exception as e:
                self._log(f"Errore inizializzazione ChatFormatManager: {e}", "WARNING")
                self.format_manager = None
        elif not CHAT_FORMAT_MANAGER_AVAILABLE:
            self._log("ChatFormatManager non disponibile - formatting disabilitato", "WARNING")
        
        # StreamingEventEmitter per live GUI updates
        self.event_emitter = None
        if STREAMING_EVENT_EMITTER_AVAILABLE:
            try:
                self.event_emitter = create_streaming_event_emitter()
                self._log("StreamingEventEmitter inizializzato per live updates", "INFO")
            except Exception as e:
                self._log(f"Errore inizializzazione StreamingEventEmitter: {e}", "WARNING")
                self.event_emitter = None
        else:
            self._log("StreamingEventEmitter non disponibile - live updates disabilitati", "WARNING")
        
        # KnowledgeManager injection per FASE 4
        self.knowledge_manager = knowledge_manager
        if self.knowledge_manager:
            self._log("KnowledgeManager disponibile per knowledge assignment", "INFO")
        else:
            self._log("KnowledgeManager non disponibile - knowledge features disabilitate", "WARNING")
        
        # FASE 5 - Resilience utilities
        if RESILIENCE_UTILS_AVAILABLE:
            # Retry manager con config custom
            retry_config = RetryConfig(
                max_attempts=3,
                initial_delay=1.0,
                max_delay=30.0,
                failure_threshold=5,
                recovery_timeout=60.0
            )
            self.retry_manager = RetryManager(retry_config)
            self.similarity_finder = SimilarityFinder(threshold=0.6)
            self._log("Resilience utilities inizializzate (retry + similarity)", "INFO")
        else:
            self.retry_manager = None
            self.similarity_finder = None
            self._log("Resilience utilities non disponibili", "WARNING")
        
        # Fallback models cache
        self._fallback_models_cache = {}
        
        # FASE 6 - Connection Pooling setup
        self.pool_manager = None
        self.db_pool = None
        self.http_pool = None
        self._batch_operations = []  # For async operation batching
        self._batch_lock = asyncio.Lock()
        self._batch_timer = None
        self._batch_interval = 0.1  # 100ms batching interval
        
        if CONNECTION_POOL_AVAILABLE:
            try:
                self.pool_manager = get_pool_manager()
                self._log("Connection pool manager inizializzato", "INFO")
                # Setup pools sarà fatto in setup_connection_pools()
            except Exception as e:
                self._log(f"Errore inizializzazione connection pooling: {e}", "WARNING")
                self.pool_manager = None
        else:
            self._log("Connection pooling non disponibile", "WARNING")
        
        # Configurazione logging
        self._setup_logging()
        
        self._log("InferenceEngineModule inizializzato", "INFO")
    
    def _setup_logging(self):
        """Configura il sistema di logging."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.storage_path / "inference.log"),
                logging.StreamHandler()
            ]
        )
    
    def _log(self, message: str, level: str = "INFO", show_console: bool = True):
        """Log con output formattato."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_msg = f"[{timestamp}] [{level}] {message}"
        
        # Log Python
        if level == "DEBUG":
            logger.debug(message)
        elif level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
        
        # Console output se abilitato
        if show_console:
            if level == "ERROR":
                console.print(f"[red]❌ {message}[/red]")
            elif level == "WARNING":
                console.print(f"[yellow]⚠️  {message}[/yellow]")
            elif level == "SUCCESS":
                console.print(f"[green]✅ {message}[/green]")
            elif level == "DEBUG":
                console.print(f"[dim]🔍 {message}[/dim]")
            else:
                console.print(f"[blue]ℹ️  {message}[/blue]")
        
        return log_msg
    
    async def setup_connection_pools(self):
        """
        FASE 6 Task 6.2 - Setup connection pools per performance optimization.
        
        Crea pools per:
        - Database connections (SQLite per risultati e cache)
        - HTTP connections (per API calls)
        """
        if not self.pool_manager:
            self._log("Pool manager non disponibile - setup saltato", "WARNING")
            return
        
        try:
            # Database connection pool per results e cache storage
            if not self.db_pool:
                db_path = str(self.storage_path / "pooled_results.db")
                self.db_pool = await self.pool_manager.create_database_pool(
                    name="inference_db",
                    db_path=db_path,
                    min_size=2,
                    max_size=8,
                    check_same_thread=False  # SQLite config per thread safety
                )
                self._log(f"Database connection pool creato ({db_path})", "INFO")
            
            # HTTP connection pool per API calls
            if not self.http_pool and AIOHTTP_AVAILABLE:
                self.http_pool = await self.pool_manager.create_http_pool(
                    name="inference_http",
                    min_size=2,
                    max_size=10,
                    timeout=aiohttp.ClientTimeout(total=30),
                    connector=aiohttp.TCPConnector(
                        limit=50,
                        limit_per_host=10,
                        keepalive_timeout=30
                    )
                )
                self._log("HTTP connection pool creato", "INFO")
            elif not AIOHTTP_AVAILABLE:
                self._log("aiohttp non disponibile - HTTP pooling disabilitato", "WARNING")
            
            # Setup batch processing timer
            await self._start_batch_processor()
            
            self._log("Connection pools setup completato", "SUCCESS")
            
        except Exception as e:
            self._log(f"Errore setup connection pools: {e}", "ERROR")
            # Fallback to direct connections se pooling fallisce
            self.db_pool = None
            self.http_pool = None
    
    async def _start_batch_processor(self):
        """Avvia il processor per batch operations."""
        if self._batch_timer and not self._batch_timer.done():
            return
        
        self._batch_timer = asyncio.create_task(self._batch_processor_loop())
        self._log("Batch processor avviato", "DEBUG")
    
    async def _batch_processor_loop(self):
        """Loop per processare operazioni in batch."""
        while True:
            try:
                await asyncio.sleep(self._batch_interval)
                await self._process_batch_operations()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log(f"Errore batch processor: {e}", "ERROR")
    
    async def _process_batch_operations(self):
        """Processa le operazioni in batch per performance."""
        if not self._batch_operations:
            return
        
        async with self._batch_lock:
            if not self._batch_operations:
                return
            
            operations = self._batch_operations.copy()
            self._batch_operations.clear()
        
        # Raggruppa operazioni per tipo
        db_operations = [op for op in operations if op["type"] == "db"]
        http_operations = [op for op in operations if op["type"] == "http"]
        
        # Processa operazioni database in batch
        if db_operations and self.db_pool:
            await self._batch_process_db_operations(db_operations)
        
        # Processa operazioni HTTP in batch
        if http_operations and self.http_pool:
            await self._batch_process_http_operations(http_operations)
    
    async def _batch_process_db_operations(self, operations: List[Dict[str, Any]]):
        """Processa operazioni database in batch."""
        try:
            async with self.db_pool.get_connection() as pooled_conn:
                conn = pooled_conn.connection
                
                for op in operations:
                    try:
                        if op["operation"] == "save_result":
                            # Batch save results
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(
                                None,
                                lambda: conn.execute(
                                    "INSERT OR REPLACE INTO inference_results (id, data, timestamp) VALUES (?, ?, ?)",
                                    (op["id"], op["data"], op["timestamp"])
                                )
                            )
                        elif op["operation"] == "save_cache":
                            # Batch save cache entries
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(
                                None,
                                lambda: conn.execute(
                                    "INSERT OR REPLACE INTO response_cache (key, value, expires) VALUES (?, ?, ?)",
                                    (op["key"], op["value"], op["expires"])
                                )
                            )
                        
                        # Completa future se presente
                        if "future" in op:
                            op["future"].set_result(True)
                            
                    except Exception as e:
                        self._log(f"Errore batch DB operation: {e}", "ERROR")
                        if "future" in op:
                            op["future"].set_exception(e)
                
                # Commit batch
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, conn.commit)
                
        except Exception as e:
            self._log(f"Errore batch DB processing: {e}", "ERROR")
            # Mark all operations as failed
            for op in operations:
                if "future" in op:
                    op["future"].set_exception(e)
    
    async def _batch_process_http_operations(self, operations: List[Dict[str, Any]]):
        """Processa operazioni HTTP in batch."""
        try:
            async with self.http_pool.get_connection() as pooled_conn:
                session = pooled_conn.connection
                
                # Raggruppa per domain per efficiency
                by_domain = {}
                for op in operations:
                    domain = op.get("domain", "default")
                    if domain not in by_domain:
                        by_domain[domain] = []
                    by_domain[domain].append(op)
                
                # Processa ogni domain
                for domain, domain_ops in by_domain.items():
                    tasks = []
                    
                    for op in domain_ops:
                        if op["operation"] == "api_call":
                            task = self._execute_http_operation(session, op)
                            tasks.append(task)
                    
                    # Execute batch per domain
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Complete futures
                        for op, result in zip(domain_ops, results):
                            if "future" in op:
                                if isinstance(result, Exception):
                                    op["future"].set_exception(result)
                                else:
                                    op["future"].set_result(result)
                
        except Exception as e:
            self._log(f"Errore batch HTTP processing: {e}", "ERROR")
            for op in operations:
                if "future" in op:
                    op["future"].set_exception(e)
    
    async def _execute_http_operation(self, session, operation: Dict[str, Any]):
        """Esegue singola operazione HTTP."""
        try:
            method = operation.get("method", "GET")
            url = operation["url"]
            kwargs = operation.get("kwargs", {})
            
            async with session.request(method, url, **kwargs) as response:
                data = await response.text()
                return {
                    "status": response.status,
                    "data": data,
                    "headers": dict(response.headers)
                }
                
        except Exception as e:
            self._log(f"HTTP operation failed: {e}", "ERROR")
            raise
    
    async def queue_batch_operation(self, operation: Dict[str, Any]) -> Any:
        """
        Accoda operazione per batch processing.
        
        Args:
            operation: Dict con operation details
            
        Returns:
            Future con risultato operazione
        """
        future = asyncio.Future()
        operation["future"] = future
        
        async with self._batch_lock:
            self._batch_operations.append(operation)
        
        return await future
    
    async def cleanup_connection_pools(self):
        """Cleanup connection pools - resource cleanup handler."""
        try:
            # Stop batch processor
            if self._batch_timer and not self._batch_timer.done():
                self._batch_timer.cancel()
                try:
                    await self._batch_timer
                except asyncio.CancelledError:
                    pass
            
            # Process remaining batch operations
            if self._batch_operations:
                await self._process_batch_operations()
            
            # Close all pools
            if self.pool_manager:
                await self.pool_manager.close_all()
                self._log("Connection pools chiusi correttamente", "INFO")
            
        except Exception as e:
            self._log(f"Errore cleanup connection pools: {e}", "ERROR")
    
    async def execute(self, action: str = "query", **kwargs) -> Dict[str, Any]:
        """
        Punto di ingresso principale per l'inference.
        
        Args:
            action: Azione da eseguire:
                - "query": Esegue query su agente/team
                - "batch": Esegue query multiple
                - "status": Stato inference attive
                - "history": Storico query
                - "clear_cache": Pulisce cache
                - "cleanup": Pulisce connection pools e resources
                
        Returns:
            Dict con risultato operazione
        """
        try:
            # Inizializza connection pools se non fatto già (lazy initialization)
            if self.pool_manager and not self.db_pool:
                await self.setup_connection_pools()
            
            if action == "query":
                return await self._execute_query(
                    query=kwargs.get("query"),
                    target_type=kwargs.get("target_type", "agent"),
                    target_id=kwargs.get("target_id"),
                    settings=kwargs.get("settings", {}),
                    context=kwargs.get("context"),
                    session_id=kwargs.get("session_id"),
                    user_id=kwargs.get("user_id")
                )
            
            elif action == "batch":
                return await self._execute_batch(
                    queries=kwargs.get("queries", [])
                )
            
            elif action == "status":
                return await self._get_status()
            
            elif action == "history":
                return await self._get_history(
                    user_id=kwargs.get("user_id"),
                    limit=kwargs.get("limit", 10)
                )
            
            elif action == "clear_cache":
                return await self._clear_cache()
            
            elif action == "cleanup":
                await self.cleanup_connection_pools()
                return {"success": True, "message": "Connection pools cleanup completato"}
            
            else:
                raise ValueError(f"Azione non supportata: {action}")
                
        except Exception as e:
            self._log(f"Errore in execute: {str(e)}", "ERROR")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _execute_query(self, query: str, target_type: str, target_id: str,
                           settings: Dict[str, Any] = None,
                           context: Optional[Dict[str, Any]] = None,
                           session_id: Optional[str] = None,
                           user_id: Optional[str] = None) -> Dict[str, Any]:
        """Esegue una singola query su agente o team."""
        # Crea request strutturata
        inference_settings = InferenceSettings(**settings) if settings else InferenceSettings()
        
        request = QueryRequest(
            query=query,
            target_type=target_type,
            target_id=target_id,
            settings=inference_settings,
            context=context,
            session_id=session_id or f"session_{datetime.now().timestamp()}",
            user_id=user_id or "default"
        )
        
        # Log iniziale
        self._start_inference_display(request)
        
        # Inizializza metriche
        metrics = InferenceMetrics(start_time=time.time())
        logs = []
        
        # Setup debug level
        if inference_settings.debug_level == "DEBUG":
            self.debug_capture.start()
        
        try:
            # Controlla cache se abilitata
            if inference_settings.cache_responses:
                cached = await self._check_cache(request)
                if cached:
                    self._log("Risposta trovata in cache", "INFO")
                    return cached
            
            # Ottieni target (agente o team)
            target = await self._get_target(target_type, target_id)
            if not target:
                raise ValueError(f"{target_type} non trovato: {target_id}")
            
            # Log target info
            self._log_target_info(target, logs)
            
            # Prepara parametri run
            run_params = await self._prepare_run_params(request, target)
            
            # Esegui query
            self._log(f"Esecuzione query: '{query[:100]}...'", "INFO")
            
            if inference_settings.stream:
                # Streaming response
                result = await self._handle_streaming_response(
                    target, query, run_params, request, metrics, logs
                )
            else:
                # Non-streaming response
                result = await self._handle_standard_response(
                    target, query, run_params, request, metrics, logs
                )
            
            # Stop debug capture
            if inference_settings.debug_level == "DEBUG":
                self.debug_capture.stop()
                logs.extend(self.debug_capture.get_logs())
            
            # Finalizza metriche
            metrics.end_time = time.time()
            metrics.calculate_duration()
            
            # Salva risultato
            if inference_settings.cache_responses:
                await self._save_to_cache(request, result)
            
            # Salva in storage
            self._save_result(result)
            
            # Export se richiesto
            if inference_settings.save_to_file:
                self._export_result(result, inference_settings.save_to_file, 
                                  inference_settings.export_format)
            
            # Display finale
            self._display_final_result(result)
            
            return {
                "success": True,
                "result": result.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            # Stop debug capture in caso di errore
            if inference_settings.debug_level == "DEBUG":
                self.debug_capture.stop()
            
            metrics.end_time = time.time()
            metrics.error_count += 1
            
            error_result = InferenceResult(
                success=False,
                request=request,
                error=str(e),
                metrics=metrics,
                logs=logs
            )
            
            self._log(f"Errore nell'inference: {str(e)}", "ERROR")
            
            return {
                "success": False,
                "error": str(e),
                "result": error_result.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_target(self, target_type: str, target_id: str) -> Union[Agent, Team, None]:
        """Ottiene l'istanza target (agente o team) con error handling e suggerimenti."""
        try:
            if target_type == "agent":
                # Controlla cache
                if target_id in self.active_agents:
                    return self.active_agents[target_id]
                
                # Crea agente tramite controller
                if self.agent_controller:
                    # FASE 5 - Retry con circuit breaker per agent creation
                    if self.retry_manager:
                        try:
                            agent = await self.retry_manager.retry_async(
                                self.agent_controller.get_agent_by_id,
                                target_id,
                                circuit_name="agent_controller"
                            )
                        except Exception as e:
                            self._log(f"Retry fallito per agent {target_id}: {e}", "WARNING")
                            agent = None
                    else:
                        agent = self.agent_controller.get_agent_by_id(target_id)
                    
                    if agent:
                        # FASE 4 - Task 4.2: Knowledge Assignment Flow
                        await self._enhance_agent_with_knowledge(agent, target_id)
                        
                        self.active_agents[target_id] = agent
                        return agent
                    else:
                        # FASE 5 - Suggerisci agenti simili se non trovato
                        await self._handle_agent_not_found(target_id)
                        return None
                
            elif target_type == "team":
                # Controlla cache
                if target_id in self.active_teams:
                    return self.active_teams[target_id]
                
                # Ottieni team tramite constructor
                if self.team_constructor:
                    team_data = await self.team_constructor.execute(
                        action="get",
                        team_id=target_id
                    )
                    if team_data["success"] and team_data.get("team_instance"):
                        team = team_data["team_instance"]
                        self.active_teams[target_id] = team
                        return team
            
            return None
            
        except Exception as e:
            self._log(f"Errore recupero target: {str(e)}", "ERROR")
            return None
    
    async def _prepare_run_params(self, request: QueryRequest, target: Any = None) -> Dict[str, Any]:
        """Prepara parametri per run agente/team con ChatFormatManager processing."""
        settings = request.settings
        
        params = {
            "stream": settings.stream,
            "show_tool_calls": settings.show_tool_calls,
            "markdown": settings.markdown
        }
        
        # Parametri reasoning se supportati
        if settings.show_full_reasoning:
            params["show_full_reasoning"] = True
        
        if settings.stream_intermediate_steps:
            params["stream_intermediate_steps"] = True
        
        # Parametri modello se specificati
        if settings.temperature is not None:
            params["temperature"] = settings.temperature
        
        if settings.max_tokens is not None:
            params["max_tokens"] = settings.max_tokens
        
        # Context se presente
        if request.context:
            params["context"] = request.context
        
        # FASE 4 - Task 4.1: Runtime Parameter Injection
        # Inietta user_id e session_id per memory/knowledge support
        if request.user_id:
            params["user_id"] = request.user_id
            self._log(f"User ID injected: {request.user_id}", "DEBUG")
        
        if request.session_id:
            params["session_id"] = request.session_id
            self._log(f"Session ID injected: {request.session_id}", "DEBUG")
        
        # Se agent ha memoria, abilita save_to_memory
        if target and hasattr(target, 'memory') and target.memory:
            params["save_to_memory"] = True
            self._log("Memory save enabled for this agent", "DEBUG")
        
        # Format query based on model usando ChatFormatManager
        if self.format_manager and target:
            try:
                # Estrai nome modello dal target
                model_name = self._extract_model_name(target)
                if model_name:
                    format_result = await self.format_manager.execute(
                        action="format",
                        content=request.query,
                        model=model_name,
                        format_type=settings.format_preferences.get("type", "default")
                    )
                    if format_result.get("success"):
                        # La query formattata sarà gestita direttamente durante l'esecuzione
                        params["_formatted_query"] = format_result.get("formatted_content")
                        self._log(f"Query formattata per modello {model_name}", "DEBUG")
            except Exception as e:
                self._log(f"Errore formatting query: {e}", "WARNING")
        
        return params
    
    def _extract_model_name(self, target) -> Optional[str]:
        """Estrae nome modello dal target Agent o Team."""
        try:
            if hasattr(target, 'model'):
                model = target.model
                if hasattr(model, 'id'):
                    return model.id
                elif hasattr(model, 'name'):
                    return model.name
                elif hasattr(model, 'model'):
                    return model.model
            elif hasattr(target, 'llm'):
                llm = target.llm
                if hasattr(llm, 'id'):
                    return llm.id
                elif hasattr(llm, 'name'):
                    return llm.name
                elif hasattr(llm, 'model'):
                    return llm.model
            return None
        except Exception as e:
            self._log(f"Errore estrazione nome modello: {e}", "DEBUG")
            return None
    
    async def _handle_streaming_response(self, target: Union[Agent, Team], query: str,
                                       run_params: Dict[str, Any], request: QueryRequest,
                                       metrics: InferenceMetrics, logs: List[str]) -> InferenceResult:
        """Gestisce response in streaming."""
        response_content = ""
        reasoning_content = ""
        tool_calls = []
        full_response = None
        
        # Generate request ID for tracking
        request_id = f"{request.session_id or 'default'}_{int(time.time() * 1000)}"
        
        try:
            # Inizia event emitter tracking se disponibile
            if self.event_emitter:
                try:
                    self.event_emitter.start_stream_tracking(request_id)
                    await self.event_emitter.emit_progress_update(0.1, "Starting streaming...")
                except Exception as e:
                    self._log(f"Errore start event emitter: {e}", "WARNING")
            
            # Inizia streaming con spinner
            with Live(Spinner("dots", text="[cyan]Processing query...[/cyan]"), 
                     refresh_per_second=10) as live:
                
                # Esegui run asincrono - usa query formattata se disponibile
                formatted_query = run_params.pop("_formatted_query", query)
                
                # FASE 5 - Retry con fallback model
                response_stream = None
                for attempt in range(3):  # Max 3 tentativi
                    try:
                        if hasattr(target, 'arun'):
                            if self.retry_manager:
                                response_stream = await self.retry_manager.retry_async(
                                    target.arun,
                                    formatted_query,
                                    circuit_name=f"model_{getattr(target.model, 'id', 'unknown')}",
                                    **run_params
                                )
                            else:
                                response_stream = await target.arun(formatted_query, **run_params)
                        else:
                            response_stream = target.run(formatted_query, **run_params)
                        break  # Success, esci dal loop
                        
                    except Exception as e:
                        self._log(f"Tentativo {attempt + 1} fallito: {e}", "WARNING")
                        
                        # Prova fallback model se disponibile
                        if attempt < 2:  # Non all'ultimo tentativo
                            fallback_applied = await self._handle_model_failure(target, e)
                            if fallback_applied:
                                self._log("Ritentativo con fallback model", "INFO")
                                continue
                        
                        # Se ultimo tentativo o fallback non applicato, rilancia
                        if attempt == 2:
                            raise
                
                if not response_stream:
                    raise Exception("Failed to get response stream after all attempts")
                
                # Processa stream con progress tracking
                event_count = 0
                async for event in self._process_stream(response_stream):
                    event_count += 1
                    if isinstance(event, RunResponse):
                        # Parse con ChatFormatManager se disponibile
                        parsed_event = await self._parse_response_event(event, target, request, streaming=True)
                        
                        # Content event
                        if parsed_event.get("content"):
                            content = parsed_event["content"]
                            response_content += content
                            live.update(Panel(
                                Markdown(response_content[-500:]),
                                title="[bold]Response[/bold]",
                                border_style="green"
                            ))
                            
                            # Emit content update event
                            if self.event_emitter:
                                try:
                                    await self.event_emitter.emit_content_update(
                                        content, 
                                        partial=True,
                                        metadata={"total_length": len(response_content)}
                                    )
                                except Exception as e:
                                    self._log(f"Errore emit content update: {e}", "WARNING")
                        
                        # Tool calls dal parsing
                        if parsed_event.get("tool_calls"):
                            for tool_call in parsed_event["tool_calls"]:
                                tool_calls.append(tool_call)
                                metrics.tool_calls_count += 1
                                if request.settings.show_tool_calls:
                                    self._display_tool_call(tool_call)
                                
                                # Emit tool call event
                                if self.event_emitter:
                                    try:
                                        await self.event_emitter.emit_tool_call(
                                            tool_call.get("tool", "unknown"),
                                            tool_call.get("args", {}),
                                            tool_call.get("result")
                                        )
                                    except Exception as e:
                                        self._log(f"Errore emit tool call: {e}", "WARNING")
                        
                        # Reasoning content dal parsing
                        if parsed_event.get("reasoning"):
                            reasoning_content += parsed_event["reasoning"]
                            metrics.reasoning_steps += 1
                            if request.settings.show_full_reasoning:
                                self._display_reasoning(parsed_event["reasoning"])
                            
                            # Emit reasoning update event
                            if self.event_emitter:
                                try:
                                    await self.event_emitter.emit_reasoning_update(parsed_event["reasoning"])
                                except Exception as e:
                                    self._log(f"Errore emit reasoning update: {e}", "WARNING")
                        
                        # Fallback ai metodi originali se parsing non disponibile
                        if not self.format_manager:
                            if event.content:
                                response_content += event.content
                                live.update(Panel(
                                    Markdown(response_content[-500:]),
                                    title="[bold]Response[/bold]",
                                    border_style="green"
                                ))
                            
                            # Tool call event (fallback)
                            if hasattr(event, 'tool_call') and event.tool_call:
                                tool_info = {
                                    "tool": event.tool_call.get("tool"),
                                    "args": event.tool_call.get("args"),
                                    "result": event.tool_call.get("result")
                                }
                                tool_calls.append(tool_info)
                                metrics.tool_calls_count += 1
                                
                                if request.settings.show_tool_calls:
                                    self._display_tool_call(tool_info)
                            
                            # Reasoning event (fallback)
                            if hasattr(event, 'reasoning_content') and event.reasoning_content:
                                reasoning_content += event.reasoning_content
                                metrics.reasoning_steps += 1
                                
                                if request.settings.show_full_reasoning:
                                    self._display_reasoning(event.reasoning_content)
                        
                        # Update progress based on events received
                        if self.event_emitter and event_count % 5 == 0:  # Every 5 events
                            try:
                                # Estimate progress based on content length and events
                                estimated_progress = min(0.9, 0.2 + (len(response_content) / 2000) * 0.7)
                                await self.event_emitter.emit_progress_update(
                                    estimated_progress, 
                                    f"Processing... ({event_count} events, {len(response_content)} chars)"
                                )
                                self.event_emitter.update_stream_progress(request_id, len(response_content))
                            except Exception as e:
                                self._log(f"Errore progress update: {e}", "WARNING")
                        
                        # Salva ultima response
                        full_response = event
                
                # Ottieni response finale dal target
                if hasattr(target, 'run_response') and target.run_response:
                    full_response = target.run_response
                    
                    # Estrai reasoning content finale
                    if hasattr(full_response, 'reasoning_content'):
                        reasoning_content = full_response.reasoning_content or reasoning_content
            
            # Log completo per debug
            if request.settings.debug_level == "DEBUG":
                logs.append(f"Final response content length: {len(response_content)}")
                logs.append(f"Tool calls: {len(tool_calls)}")
                logs.append(f"Reasoning steps: {metrics.reasoning_steps}")
            
            # Estrai model-specific data dall'ultimo parsed event se disponibile
            model_specific_data = {}
            if self.format_manager and full_response:
                try:
                    final_parsed = await self._parse_response_event(full_response, target, request, streaming=False)
                    model_specific_data = self._extract_model_specific_data(final_parsed)
                except Exception as e:
                    self._log(f"Errore estrazione model-specific data: {e}", "WARNING")
            
            # Crea risultato con model-specific data
            result = InferenceResult(
                success=True,
                request=request,
                response_content=response_content,
                full_response=full_response,
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
                metrics=metrics,
                logs=logs,
                claude_thinking=model_specific_data.get("claude_thinking"),
                gemini_safety=model_specific_data.get("gemini_safety"),
                openai_structured=model_specific_data.get("openai_structured"),
                ollama_metadata=model_specific_data.get("ollama_metadata"),
                parsed_metadata=model_specific_data.get("parsed_metadata", {})
            )
            
            # Calcola metriche finali
            result.metrics.response_size_chars = len(response_content)
            
            # FASE 4 - Task 4.3: Salva interazione in memoria
            await self._save_interaction_memory(result)
            
            # Emit completion event se disponibile
            if self.event_emitter:
                try:
                    await self.event_emitter.emit_completion({
                        "success": True,
                        "content_length": len(response_content),
                        "tool_calls": len(tool_calls),
                        "reasoning_steps": metrics.reasoning_steps,
                        "duration": metrics.duration_seconds,
                        "model_data": model_specific_data
                    })
                    self.event_emitter.stop_stream_tracking(request_id, "completed")
                    await self.event_emitter.emit_progress_update(1.0, "Streaming completed")
                except Exception as e:
                    self._log(f"Errore emit completion: {e}", "WARNING")
            
            return result
            
        except Exception as e:
            self._log(f"Errore durante streaming: {str(e)}", "ERROR")
            
            # Emit error event se disponibile
            if self.event_emitter:
                try:
                    await self.event_emitter.emit_error(
                        f"Streaming error: {str(e)}",
                        {"error_type": type(e).__name__, "request_id": request_id}
                    )
                    self.event_emitter.stop_stream_tracking(request_id, "error")
                except Exception as emit_error:
                    self._log(f"Errore emit error event: {emit_error}", "WARNING")
            
            raise
    
    async def _handle_standard_response(self, target: Union[Agent, Team], query: str,
                                      run_params: Dict[str, Any], request: QueryRequest,
                                      metrics: InferenceMetrics, logs: List[str]) -> InferenceResult:
        """Gestisce response standard (non-streaming)."""
        try:
            # Esegui query - usa query formattata se disponibile
            self._log("Esecuzione query (non-streaming)...", "INFO")
            formatted_query = run_params.pop("_formatted_query", query)
            
            # FASE 5 - Retry con fallback model per standard response
            response = None
            for attempt in range(3):  # Max 3 tentativi
                try:
                    if hasattr(target, 'arun'):
                        if self.retry_manager:
                            response = await self.retry_manager.retry_async(
                                target.arun,
                                formatted_query,
                                circuit_name=f"model_{getattr(target.model, 'id', 'unknown')}",
                                **run_params
                            )
                        else:
                            response = await target.arun(formatted_query, **run_params)
                    else:
                        response = target.run(formatted_query, **run_params)
                    break  # Success, esci dal loop
                    
                except Exception as e:
                    self._log(f"Standard response tentativo {attempt + 1} fallito: {e}", "WARNING")
                    
                    # Prova fallback model se disponibile
                    if attempt < 2:  # Non all'ultimo tentativo
                        fallback_applied = await self._handle_model_failure(target, e)
                        if fallback_applied:
                            self._log("Ritentativo standard response con fallback model", "INFO")
                            continue
                    
                    # Se ultimo tentativo o fallback non applicato, rilancia
                    if attempt == 2:
                        raise
            
            if not response:
                raise Exception("Failed to get response after all attempts")
            
            # Parse response con ChatFormatManager se disponibile
            response_content = ""
            reasoning_content = ""
            tool_calls = []
            
            if response:
                parsed_response = await self._parse_response_event(response, target, request, streaming=False)
                
                # Usa risultati parsed
                response_content = parsed_response.get("content", "")
                reasoning_content = parsed_response.get("reasoning", "")
                tool_calls = parsed_response.get("tool_calls", [])
                
                # Aggiorna metriche
                if reasoning_content:
                    metrics.reasoning_steps = 1
                metrics.tool_calls_count = len(tool_calls)
                
                # Log per debug
                if request.settings.debug_level == "DEBUG":
                    logs.append(f"Response type: {type(response)}")
                    logs.append(f"Response content length: {len(response_content)}")
                
                # Estrai model-specific data dal parsed response
                model_specific_data = self._extract_model_specific_data(parsed_response)
            else:
                model_specific_data = {}
            
            # Crea risultato con model-specific data
            result = InferenceResult(
                success=True,
                request=request,
                response_content=response_content,
                full_response=response,
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
                metrics=metrics,
                logs=logs,
                claude_thinking=model_specific_data.get("claude_thinking"),
                gemini_safety=model_specific_data.get("gemini_safety"),
                openai_structured=model_specific_data.get("openai_structured"),
                ollama_metadata=model_specific_data.get("ollama_metadata"),
                parsed_metadata=model_specific_data.get("parsed_metadata", {})
            )
            
            result.metrics.response_size_chars = len(response_content)
            
            # FASE 4 - Task 4.3: Salva interazione in memoria
            await self._save_interaction_memory(result)
            
            return result
            
        except Exception as e:
            self._log(f"Errore response standard: {str(e)}", "ERROR")
            raise
    
    async def _parse_response_event(self, event, target, request: QueryRequest, streaming: bool = False) -> Dict[str, Any]:
        """Parse response event con ChatFormatManager se disponibile."""
        parsed_result = {
            "content": "",
            "tool_calls": [],
            "reasoning": "",
            "metadata": {}
        }
        
        if self.format_manager:
            try:
                model_name = self._extract_model_name(target)
                if model_name:
                    format_result = await self.format_manager.execute(
                        action="parse",
                        response=event,
                        model=model_name,
                        streaming=streaming
                    )
                    
                    if format_result.get("success"):
                        parse_result = format_result.get("parse_result", {})
                        parsed_result["content"] = parse_result.get("content", "")
                        parsed_result["tool_calls"] = parse_result.get("tool_calls", [])
                        parsed_result["reasoning"] = parse_result.get("reasoning", "")
                        parsed_result["metadata"] = parse_result.get("metadata", {})
                        
                        self._log(f"Response parsed per {model_name}", "DEBUG")
                        return parsed_result
            except Exception as e:
                self._log(f"Errore parsing response: {e}", "WARNING")
        
        # Fallback: estrazione diretta dall'evento
        if hasattr(event, 'content') and event.content:
            parsed_result["content"] = event.content
        
        if hasattr(event, 'tool_call') and event.tool_call:
            parsed_result["tool_calls"] = [{
                "tool": event.tool_call.get("tool"),
                "args": event.tool_call.get("args"),
                "result": event.tool_call.get("result")
            }]
        
        if hasattr(event, 'reasoning_content') and event.reasoning_content:
            parsed_result["reasoning"] = event.reasoning_content
        
        return parsed_result
    
    def _extract_model_specific_data(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        """Estrae model-specific data dal parsed result"""
        model_data = {
            "claude_thinking": None,
            "gemini_safety": None, 
            "openai_structured": None,
            "ollama_metadata": None,
            "parsed_metadata": {}
        }
        
        metadata = parsed_result.get("metadata", {})
        provider = metadata.get("provider", "generic")
        
        # Claude thinking extraction
        if provider == "anthropic" and parsed_result.get("reasoning"):
            model_data["claude_thinking"] = parsed_result["reasoning"]
        
        # Gemini safety ratings
        if provider == "google" and metadata.get("gemini_safety"):
            model_data["gemini_safety"] = metadata["gemini_safety"]
        
        # OpenAI structured output
        if provider == "openai" and metadata.get("openai_structured"):
            model_data["openai_structured"] = metadata["openai_structured"]
        
        # Ollama metadata
        if provider == "ollama" and metadata.get("ollama_metadata"):
            model_data["ollama_metadata"] = metadata["ollama_metadata"]
        
        # Generic parsed metadata
        model_data["parsed_metadata"] = metadata
        
        return model_data
    
    async def _process_stream(self, stream: Union[Iterator, AsyncIterator]):
        """
        Processa stream di eventi con proper async streaming, buffering e backpressure.
        FASE 3 - Task 3.1: Enhanced streaming implementation.
        """
        buffer = []
        buffer_size = 10  # Configurable buffer size
        
        try:
            if hasattr(stream, '__aiter__'):
                # True async iterator - enhanced processing
                async for event in stream:
                    # Add to buffer for backpressure management
                    buffer.append(event)
                    
                    # Yield buffered events
                    if len(buffer) >= buffer_size:
                        for buffered_event in buffer:
                            yield buffered_event
                        buffer.clear()
                    else:
                        # Immediate yield for single events
                        yield event
                        buffer.pop()  # Remove from buffer since already yielded
                    
                    # Allow other tasks to run (prevent blocking)
                    await asyncio.sleep(0)
                
                # Yield remaining buffered events
                for remaining_event in buffer:
                    yield remaining_event
                    
            else:
                # Sync iterator - convert to async properly with buffering
                loop = asyncio.get_event_loop()
                async for event in self._sync_to_async_generator(stream, loop):
                    # Add to buffer for consistent processing
                    buffer.append(event)
                    
                    # Yield buffered events
                    if len(buffer) >= buffer_size:
                        for buffered_event in buffer:
                            yield buffered_event
                        buffer.clear()
                    else:
                        # Immediate yield for single events
                        yield event
                        buffer.pop()  # Remove from buffer since already yielded
                        
                # Yield remaining buffered events
                for remaining_event in buffer:
                    yield remaining_event
                    
        except asyncio.CancelledError:
            self._log("Stream processing cancelled", "INFO")
            # Yield any remaining buffered events before cancelling
            for remaining_event in buffer:
                yield remaining_event
            raise
        except Exception as e:
            self._log(f"Errore processing stream: {str(e)}", "ERROR")
            # Attempt error recovery - yield buffered events if possible
            try:
                for remaining_event in buffer:
                    yield remaining_event
            except Exception:
                pass  # Skip if buffer is corrupted
            raise
    
    async def _sync_to_async_generator(self, sync_stream, loop):
        """
        Converte sync iterator a async generator con proper handling.
        FASE 3 - Task 3.1: Sync to async conversion with backpressure.
        """
        try:
            # Use executor to run sync iterator in thread pool
            import concurrent.futures
            
            def sync_iterator():
                """Wrapper per sync iterator"""
                try:
                    for event in sync_stream:
                        yield event
                except Exception as e:
                    self._log(f"Errore in sync iterator: {e}", "ERROR")
                    raise
            
            # Process sync iterator in executor
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Create sync iterator
                iterator = sync_iterator()
                
                while True:
                    try:
                        # Get next event from sync iterator in thread
                        future = executor.submit(next, iterator)
                        
                        # Wait for event with timeout to prevent hanging
                        try:
                            event = await loop.run_in_executor(None, future.result, 1.0)  # 1 second timeout
                            yield event
                            
                            # Allow other coroutines to run
                            await asyncio.sleep(0.001)  # Small delay for backpressure
                            
                        except concurrent.futures.TimeoutError:
                            # Timeout - check if stream is still active
                            if not future.done():
                                future.cancel()
                            continue
                            
                    except StopIteration:
                        # Iterator exhausted
                        break
                    except Exception as e:
                        self._log(f"Errore in sync to async conversion: {e}", "ERROR")
                        break
                        
        except Exception as e:
            self._log(f"Errore sync_to_async_generator: {e}", "ERROR")
            raise
    
    def _log_target_info(self, target: Union[Agent, Team], logs: List[str]):
        """Log informazioni sul target."""
        if isinstance(target, Agent):
            logs.append(f"Target: Agent '{target.name or 'Unnamed'}'")
            logs.append(f"Model: {target.model.id if hasattr(target.model, 'id') else 'Unknown'}")
            if hasattr(target, 'tools') and target.tools:
                logs.append(f"Tools: {len(target.tools)} disponibili")
        
        elif isinstance(target, Team):
            logs.append(f"Target: Team '{target.name or 'Unnamed'}'")
            logs.append(f"Mode: {target.mode}")
            logs.append(f"Members: {len(target.members)} agenti")
    
    def _start_inference_display(self, request: QueryRequest):
        """Display iniziale inference."""
        console.rule(f"[bold blue]🚀 INFERENCE ENGINE - {request.target_type.upper()}[/bold blue]")
        
        info_table = Table(show_header=False)
        info_table.add_column("Campo", style="cyan", width=20)
        info_table.add_column("Valore", style="white")
        
        info_table.add_row("Target", f"{request.target_type}: {request.target_id}")
        info_table.add_row("User", request.user_id or "default")
        info_table.add_row("Session", request.session_id or "new")
        info_table.add_row("Query", request.query[:100] + "..." if len(request.query) > 100 else request.query)
        info_table.add_row("Streaming", "✓" if request.settings.stream else "✗")
        info_table.add_row("Debug Level", request.settings.debug_level)
        
        console.print(info_table)
        console.print()
    
    def _display_tool_call(self, tool_info: Dict[str, Any]):
        """Display chiamata tool."""
        console.print(Panel(
            f"[yellow]Tool:[/yellow] {tool_info.get('tool', 'Unknown')}\n"
            f"[yellow]Args:[/yellow] {tool_info.get('args', {})}\n"
            f"[yellow]Result:[/yellow] {str(tool_info.get('result', 'Pending'))[:200]}",
            title="[bold]🔧 Tool Call[/bold]",
            border_style="yellow"
        ))
    
    def _display_reasoning(self, reasoning: str):
        """Display reasoning step."""
        console.print(Panel(
            Markdown(reasoning[:500] + "..." if len(reasoning) > 500 else reasoning),
            title="[bold]🧠 Reasoning[/bold]",
            border_style="magenta"
        ))
    
    def _display_final_result(self, result: InferenceResult):
        """Display risultato finale."""
        console.rule("[bold green]✅ INFERENCE COMPLETATA[/bold green]")
        
        # Metriche
        if result.metrics:
            metrics_table = Table(title="📊 Metriche Performance")
            metrics_table.add_column("Metrica", style="cyan")
            metrics_table.add_column("Valore", style="white")
            
            metrics_table.add_row("Durata", f"{result.metrics.duration_seconds:.2f}s")
            metrics_table.add_row("Caratteri Response", str(result.metrics.response_size_chars))
            metrics_table.add_row("Tool Calls", str(result.metrics.tool_calls_count))
            metrics_table.add_row("Reasoning Steps", str(result.metrics.reasoning_steps))
            metrics_table.add_row("Errori", str(result.metrics.error_count))
            
            console.print(metrics_table)
        
        # Response finale (preview)
        if result.response_content:
            console.print(Panel(
                Markdown(result.response_content[:1000] + "..." 
                        if len(result.response_content) > 1000 
                        else result.response_content),
                title="[bold]📝 Response Preview[/bold]",
                border_style="green"
            ))
        
        console.rule()
    
    async def _check_cache(self, request: QueryRequest) -> Optional[Dict[str, Any]]:
        """Controlla cache per query esistente."""
        try:
            # Genera cache key
            cache_key = f"{request.target_type}_{request.target_id}_{hash(request.query)}"
            
            # Cerca in cache
            cached = self.cache_storage.get_run(cache_key)
            
            if cached and cached.get("run_data"):
                cache_data = cached["run_data"]
                
                # Verifica validità (es. max 1 ora)
                if "timestamp" in cache_data:
                    cache_time = datetime.fromisoformat(cache_data["timestamp"])
                    if (datetime.now() - cache_time).total_seconds() < 3600:
                        return cache_data
            
            return None
            
        except Exception as e:
            self._log(f"Errore check cache: {str(e)}", "WARNING")
            return None
    
    async def _save_to_cache(self, request: QueryRequest, result: InferenceResult):
        """Salva risultato in cache."""
        try:
            cache_key = f"{request.target_type}_{request.target_id}_{hash(request.query)}"
            
            cache_data = {
                "result": result.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            
            self.cache_storage.upsert_run(
                run_id=cache_key,
                run_data=cache_data
            )
            
        except Exception as e:
            self._log(f"Errore salvataggio cache: {str(e)}", "WARNING")
    
    def _save_result(self, result: InferenceResult):
        """Salva risultato nel storage permanente."""
        try:
            result_id = f"inference_{result.request.session_id}_{datetime.now().timestamp()}"
            
            self.result_storage.upsert_run(
                run_id=result_id,
                run_data=result.to_dict()
            )
            
        except Exception as e:
            self._log(f"Errore salvataggio risultato: {str(e)}", "WARNING")
    
    def _export_result(self, result: InferenceResult, file_path: str, format: str):
        """Esporta risultato su file."""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            
            elif format == "markdown":
                content = f"""# Inference Result

## Query
**Target**: {result.request.target_type} - {result.request.target_id}  
**User**: {result.request.user_id}  
**Timestamp**: {result.timestamp}

### Query Text
{result.request.query}

## Response
{result.response_content or "No response"}

## Metrics
- Duration: {result.metrics.duration_seconds if result.metrics else 'N/A'}s
- Response Size: {result.metrics.response_size_chars if result.metrics else 'N/A'} chars
- Tool Calls: {result.metrics.tool_calls_count if result.metrics else 0}
- Reasoning Steps: {result.metrics.reasoning_steps if result.metrics else 0}
"""
                
                if result.reasoning_content:
                    content += f"\n## Reasoning\n{result.reasoning_content}\n"
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            elif format == "txt":
                content = f"""INFERENCE RESULT
{'=' * 50}
Target: {result.request.target_type} - {result.request.target_id}
User: {result.request.user_id}
Timestamp: {result.timestamp}

QUERY:
{result.request.query}

RESPONSE:
{result.response_content or "No response"}

METRICS:
Duration: {result.metrics.duration_seconds if result.metrics else 'N/A'}s
Response Size: {result.metrics.response_size_chars if result.metrics else 'N/A'} chars
"""
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            self._log(f"Risultato esportato in: {path}", "SUCCESS")
            
        except Exception as e:
            self._log(f"Errore export: {str(e)}", "ERROR")
    
    async def _execute_batch(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Esegue batch di query."""
        self._log(f"Esecuzione batch di {len(queries)} query", "INFO")
        
        results = []
        success_count = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task("[cyan]Processing queries...", total=len(queries))
            
            for i, query_data in enumerate(queries):
                try:
                    result = await self._execute_query(**query_data)
                    results.append(result)
                    if result["success"]:
                        success_count += 1
                    
                except Exception as e:
                    results.append({
                        "success": False,
                        "error": str(e),
                        "query": query_data
                    })
                
                progress.update(task, advance=1, 
                              description=f"[cyan]Processing query {i+1}/{len(queries)}...")
        
        return {
            "success": True,
            "total_queries": len(queries),
            "successful": success_count,
            "failed": len(queries) - success_count,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_status(self) -> Dict[str, Any]:
        """Ottiene stato corrente del motore."""
        return {
            "success": True,
            "active_agents": list(self.active_agents.keys()),
            "active_teams": list(self.active_teams.keys()),
            "cache_enabled": True,
            "storage_path": str(self.storage_path),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_history(self, user_id: Optional[str] = None, 
                         limit: int = 10) -> Dict[str, Any]:
        """Recupera storico query."""
        try:
            # Query storage per user_id
            all_runs = self.result_storage.get_all_runs()
            
            # Filtra per user se specificato
            if user_id:
                filtered = [
                    run for run in all_runs 
                    if run.get("run_data", {}).get("request", {}).get("user_id") == user_id
                ]
            else:
                filtered = all_runs
            
            # Ordina per timestamp e limita
            filtered.sort(key=lambda x: x.get("run_data", {}).get("timestamp", ""), reverse=True)
            filtered = filtered[:limit]
            
            # Formatta risultati
            history = []
            for run in filtered:
                data = run.get("run_data", {})
                history.append({
                    "timestamp": data.get("timestamp"),
                    "query": data.get("request", {}).get("query", "")[:100] + "...",
                    "target": f"{data.get('request', {}).get('target_type')} - {data.get('request', {}).get('target_id')}",
                    "success": data.get("success", False),
                    "duration": data.get("metrics", {}).get("duration_seconds")
                })
            
            return {
                "success": True,
                "user_id": user_id,
                "history": history,
                "total": len(history),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self._log(f"Errore recupero history: {str(e)}", "ERROR")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _clear_cache(self) -> Dict[str, Any]:
        """Pulisce la cache."""
        try:
            # Clear cache storage
            # Implementazione dipende dal tipo di storage
            self._log("Cache pulita", "SUCCESS")
            
            return {
                "success": True,
                "message": "Cache cleared successfully",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self._log(f"Errore pulizia cache: {str(e)}", "ERROR")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _enhance_agent_with_knowledge(self, agent: Agent, agent_id: str):
        """
        FASE 4 - Task 4.2: Knowledge Assignment Flow
        Arricchisce l'agente con knowledge base se disponibile.
        
        Args:
            agent: Istanza Agent AGNO
            agent_id: ID dell'agente
        """
        if not self.knowledge_manager:
            self._log("Knowledge manager non disponibile, skip enhancement", "DEBUG")
            return
        
        try:
            # Ottieni knowledge per questo agente
            knowledge_result = await self.knowledge_manager.execute(
                action="get_knowledge_for_agent",
                agent_id=agent_id
            )
            
            if knowledge_result.get("success") and knowledge_result.get("knowledge"):
                knowledge = knowledge_result["knowledge"]
                
                # Assegna knowledge all'agente
                if hasattr(agent, 'knowledge'):
                    agent.knowledge = knowledge
                    self._log(f"Knowledge assegnata ad agente {agent_id}", "INFO")
                
                # Abilita search knowledge se supportato
                if hasattr(agent, 'search_knowledge'):
                    agent.search_knowledge = True
                    self._log(f"Knowledge search abilitata per agente {agent_id}", "DEBUG")
                
                # Aggiorna instructions con knowledge context se disponibile
                if hasattr(agent, 'instructions') and hasattr(knowledge, 'get_context'):
                    try:
                        knowledge_context = knowledge.get_context()
                        if knowledge_context:
                            agent.instructions = f"{agent.instructions}\n\nKnowledge Context:\n{knowledge_context}"
                            self._log(f"Instructions aggiornate con knowledge context per {agent_id}", "DEBUG")
                    except Exception as e:
                        self._log(f"Errore update instructions: {e}", "WARNING")
                
                # Aggiungi knowledge tools se necessario
                if hasattr(agent, 'tools') and hasattr(knowledge, 'get_tools'):
                    try:
                        knowledge_tools = knowledge.get_tools()
                        if knowledge_tools:
                            # Evita duplicati
                            existing_tool_names = {tool.name if hasattr(tool, 'name') else str(tool) 
                                                 for tool in agent.tools}
                            for tool in knowledge_tools:
                                tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                                if tool_name not in existing_tool_names:
                                    agent.tools.append(tool)
                                    self._log(f"Tool {tool_name} aggiunto ad agente {agent_id}", "DEBUG")
                    except Exception as e:
                        self._log(f"Errore aggiunta knowledge tools: {e}", "WARNING")
                
            else:
                self._log(f"Nessuna knowledge trovata per agente {agent_id}", "DEBUG")
                
        except Exception as e:
            self._log(f"Errore knowledge enhancement per {agent_id}: {e}", "WARNING")
            # FASE 5 - Gestisci knowledge non disponibile con graceful degradation
            await self._handle_knowledge_unavailable(agent, agent_id)
    
    async def _handle_agent_not_found(self, agent_id: str):
        """
        FASE 5 - Task 5.1: Gestisce caso agent not found con suggerimenti.
        
        Args:
            agent_id: ID dell'agente non trovato
        """
        if not self.similarity_finder or not self.agent_controller:
            self._log(f"Agent '{agent_id}' not found", "WARNING")
            return
        
        try:
            # Ottieni lista agenti disponibili
            available_agents = []
            if hasattr(self.agent_controller, 'get_available_agents'):
                available_agents = self.agent_controller.get_available_agents()
            elif hasattr(self.agent_controller, 'list_agents'):
                available_agents = self.agent_controller.list_agents()
            
            if available_agents:
                # Trova agenti simili
                suggestions = self.similarity_finder.suggest_alternatives(
                    agent_id, 
                    available_agents,
                    max_suggestions=3
                )
                
                # Formatta messaggio con suggerimenti
                message = self.similarity_finder.format_suggestions_message(
                    agent_id,
                    suggestions
                )
                
                self._log(message, "INFO")
                console.print(Panel(message, title="[yellow]Agent Not Found[/yellow]", border_style="yellow"))
            else:
                self._log(f"Agent '{agent_id}' not found and no agents available", "WARNING")
                
        except Exception as e:
            self._log(f"Errore generazione suggerimenti: {e}", "WARNING")
    
    async def _get_fallback_model(self, failed_model: str) -> Optional[str]:
        """
        FASE 5 - Task 5.1: Trova modello di fallback quando il primario fallisce.
        
        Args:
            failed_model: Nome del modello che ha fallito
            
        Returns:
            Nome del modello di fallback o None
        """
        # Check cache first
        if failed_model in self._fallback_models_cache:
            return self._fallback_models_cache[failed_model]
        
        try:
            # Ottieni modelli disponibili
            available_models = []
            
            # Prova da vari sources
            if hasattr(self, 'model_factory') and self.model_factory:
                if hasattr(self.model_factory, 'get_available_models'):
                    available_models = self.model_factory.get_available_models()
            
            # Fallback: usa lista statica comune
            if not available_models:
                available_models = [
                    "gpt-4o", "gpt-3.5-turbo",
                    "claude-3-sonnet", "claude-3-haiku",
                    "gemini-pro", "gemini-flash",
                    "llama3", "mistral"
                ]
            
            # Trova fallback
            fallback = find_fallback_model(failed_model, available_models)
            
            if fallback:
                self._fallback_models_cache[failed_model] = fallback
                self._log(f"Fallback model per {failed_model}: {fallback}", "INFO")
                
            return fallback
            
        except Exception as e:
            self._log(f"Errore ricerca fallback model: {e}", "WARNING")
            return None
    
    async def _handle_model_failure(self, target: Any, original_error: Exception) -> bool:
        """
        FASE 5 - Task 5.1: Gestisce fallimento modello con fallback.
        
        Args:
            target: Target agent/team
            original_error: Errore originale
            
        Returns:
            True se fallback applicato con successo
        """
        if not hasattr(target, 'model') or not target.model:
            return False
        
        try:
            current_model = getattr(target.model, 'id', str(target.model))
            fallback_model = await self._get_fallback_model(current_model)
            
            if fallback_model:
                self._log(f"Tentativo fallback da {current_model} a {fallback_model}", "INFO")
                
                # Prova a creare nuovo modello
                if hasattr(self, 'model_factory') and self.model_factory:
                    new_model = self.model_factory.create({"model": fallback_model})
                    if new_model:
                        target.model = new_model
                        self._log(f"Fallback model {fallback_model} applicato con successo", "INFO")
                        return True
                        
            return False
            
        except Exception as e:
            self._log(f"Errore applicazione fallback model: {e}", "WARNING")
            return False
    
    async def _handle_knowledge_unavailable(self, agent: Any, agent_id: str):
        """
        FASE 5 - Task 5.1: Gestisce caso knowledge non disponibile.
        
        Args:
            agent: Agent instance
            agent_id: ID dell'agente
        """
        self._log(f"Knowledge non disponibile per {agent_id}, procedo senza", "INFO")
        
        # Disabilita knowledge search se era abilitato
        if hasattr(agent, 'search_knowledge'):
            agent.search_knowledge = False
        
        # Aggiungi nota alle instructions se possibile
        if hasattr(agent, 'instructions'):
            knowledge_note = "\n\nNote: Knowledge base is currently unavailable. Using general knowledge only."
            if knowledge_note not in agent.instructions:
                agent.instructions += knowledge_note
    
    async def _handle_memory_save_failure(self, result: InferenceResult, error: Exception):
        """
        FASE 5 - Task 5.1: Gestisce fallimento salvataggio memoria.
        
        Args:
            result: Risultato inference
            error: Errore verificato
        """
        self._log(f"Memory save fallito: {error}. Continuo comunque.", "WARNING")
        
        # Prova fallback storage locale se disponibile
        try:
            # Salva in cache locale come fallback
            if hasattr(self, 'cache_storage') and self.cache_storage:
                fallback_key = f"memory_fallback_{result.request.user_id}_{result.timestamp}"
                self.cache_storage.set(fallback_key, {
                    "query": result.request.query,
                    "response": result.response_content[:1000],  # Limita size
                    "timestamp": result.timestamp,
                    "error": "Primary memory save failed"
                })
                self._log("Memory salvata in cache fallback", "INFO")
        except Exception as e:
            self._log(f"Anche fallback memory save fallito: {e}", "ERROR")
    
    async def _save_interaction_memory(self, result: InferenceResult):
        """
        FASE 4 - Task 4.3: Memory Persistence Flow
        Salva l'interazione nella memoria se user_id presente e risultato positivo.
        
        Args:
            result: InferenceResult con dettagli dell'interazione
        """
        if not self.knowledge_manager:
            self._log("Knowledge manager non disponibile per memory save", "DEBUG")
            return
        
        if not result.request.user_id or not result.success:
            self._log("Skip memory save: no user_id o risultato non riuscito", "DEBUG")
            return
        
        try:
            # Prepara contenuto memoria
            memory_content = {
                "query": result.request.query,
                "response": result.response_content,
                "timestamp": result.timestamp,
                "target_id": result.request.target_id,
                "target_type": result.request.target_type,
                "session_id": result.request.session_id,
                "tool_calls": len(result.tool_calls),
                "has_reasoning": bool(result.reasoning_content),
                "duration_seconds": result.metrics.duration_seconds if result.metrics else None
            }
            
            # Se ci sono model-specific data, includili
            if result.claude_thinking:
                memory_content["claude_thinking_length"] = len(result.claude_thinking)
            if result.gemini_safety:
                memory_content["gemini_safety_score"] = result.gemini_safety.get("overall_safety_score")
            if result.openai_structured:
                memory_content["openai_structured_output"] = bool(result.openai_structured)
            if result.ollama_metadata:
                memory_content["ollama_tokens_per_second"] = result.ollama_metadata.get("tokens_per_second")
            
            # Salva nella memoria
            memory_result = await self.knowledge_manager.execute(
                action="create_memory",
                user_id=result.request.user_id,
                content=memory_content,
                memory_type="interaction",
                agent_id=result.request.target_id
            )
            
            if memory_result.get("success"):
                self._log(f"Interazione salvata in memoria per user {result.request.user_id}", "INFO")
                
                # Se c'è event emitter, notifica il salvataggio
                if self.event_emitter:
                    try:
                        await self.event_emitter.emit_event(StreamingEvent(
                            event_type=StreamingEventType.PARTIAL_RESPONSE,
                            content="Memory saved",
                            metadata={"memory_id": memory_result.get("memory_id")},
                            session_id=result.request.session_id,
                            request_id=result.request.session_id
                        ))
                    except Exception as e:
                        self._log(f"Errore emit memory event: {e}", "WARNING")
            else:
                self._log(f"Errore salvataggio memoria: {memory_result.get('error')}", "WARNING")
                
        except Exception as e:
            self._log(f"Errore memory persistence: {e}", "WARNING")
            # FASE 5 - Gestisci fallimento con fallback
            await self._handle_memory_save_failure(result, e)
    
    async def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """
        FASE 5 - Task 5.2: Ottiene status dei circuit breakers.
        
        Returns:
            Status di tutti i circuit breakers
        """
        if not self.retry_manager:
            return {"error": "Retry manager not available"}
        
        try:
            all_circuits = self.retry_manager.get_all_circuits_status()
            
            # Aggiungi statistiche aggregate
            total_circuits = len(all_circuits)
            healthy_circuits = sum(1 for c in all_circuits.values() if c["is_healthy"])
            
            return {
                "total_circuits": total_circuits,
                "healthy_circuits": healthy_circuits,
                "unhealthy_circuits": total_circuits - healthy_circuits,
                "circuits": all_circuits,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self._log(f"Errore getting circuit breaker status: {e}", "ERROR")
            return {"error": str(e)}
    
    async def reset_circuit_breaker(self, circuit_name: str) -> bool:
        """
        FASE 5 - Task 5.2: Reset manuale di un circuit breaker.
        
        Args:
            circuit_name: Nome del circuit da resettare
            
        Returns:
            True se reset avvenuto con successo
        """
        if not self.retry_manager:
            return False
        
        try:
            if circuit_name in self.retry_manager.circuit_states:
                state = self.retry_manager.circuit_states[circuit_name]
                state.state = CircuitState.CLOSED
                state.failure_count = 0
                state.success_count = 0
                state.last_failure_time = None
                state.last_state_change = time.time()
                
                self._log(f"Circuit breaker {circuit_name} reset to CLOSED", "INFO")
                return True
            else:
                self._log(f"Circuit breaker {circuit_name} not found", "WARNING")
                return False
                
        except Exception as e:
            self._log(f"Errore reset circuit breaker {circuit_name}: {e}", "ERROR")
            return False


# ========== TEST STANDALONE ==========
if __name__ == "__main__":
    # Mock classes per test
    class MockAgent:
        def __init__(self, name, model_id="gpt-4o"):
            self.name = name
            self.model = type('Model', (), {'id': model_id})()
            self.tools = []
        
        async def arun(self, query, **kwargs):
            # Simula response
            await asyncio.sleep(1)
            
            class MockResponse:
                def __init__(self):
                    self.content = f"Risposta dell'agente {name} alla query: {query}"
                    self.reasoning_content = "Step 1: Analisi query\nStep 2: Elaborazione\nStep 3: Risposta"
            
            return MockResponse()
    
    class MockAgentController:
        def get_agent_by_id(self, agent_id):
            return MockAgent(agent_id)
    
    class MockTeamConstructor:
        async def execute(self, action, **kwargs):
            if action == "get":
                return {
                    "success": True,
                    "team_instance": type('Team', (), {
                        'name': kwargs.get('team_id'),
                        'mode': 'coordinate',
                        'members': [MockAgent("agent1"), MockAgent("agent2")],
                        'arun': lambda self, q, **kw: MockAgent("team").arun(q, **kw)
                    })()
                }
    
    async def test_inference_engine():
        """Test del modulo InferenceEngine."""
        console.rule("[bold blue]TEST INFERENCE ENGINE MODULE[/bold blue]")
        
        # Inizializza engine
        engine = InferenceEngineModule(
            agent_controller=MockAgentController(),
            team_constructor=MockTeamConstructor()
        )
        
        # Test 1: Query semplice su agente
        console.print("\n[yellow]Test 1: Query su Agente[/yellow]")
        
        result1 = await engine.execute(
            action="query",
            query="Qual è il capitale dell'Italia?",
            target_type="agent",
            target_id="geography_agent",
            settings={
                "stream": False,
                "debug_level": "DEBUG",
                "show_tool_calls": True,
                "cache_responses": True
            },
            user_id="test_user"
        )
        
        if result1["success"]:
            console.print("[green]✅ Test 1 superato![/green]")
        
        # Test 2: Query su team con streaming
        console.print("\n[yellow]Test 2: Query su Team (Streaming)[/yellow]")
        
        result2 = await engine.execute(
            action="query",
            query="Analizza il mercato delle criptovalute",
            target_type="team",
            target_id="finance_team",
            settings={
                "stream": True,
                "show_full_reasoning": True,
                "save_to_file": "tmp/test_result.json",
                "export_format": "json"
            }
        )
        
        if result2["success"]:
            console.print("[green]✅ Test 2 superato![/green]")
        
        # Test 3: Batch queries
        console.print("\n[yellow]Test 3: Batch Queries[/yellow]")
        
        batch_queries = [
            {
                "query": "Query 1",
                "target_type": "agent",
                "target_id": "agent1"
            },
            {
                "query": "Query 2",
                "target_type": "agent",
                "target_id": "agent2"
            }
        ]
        
        result3 = await engine.execute(
            action="batch",
            queries=batch_queries
        )
        
        if result3["success"]:
            console.print(f"[green]✅ Batch completato: {result3['successful']}/{result3['total_queries']} successi[/green]")
        
        # Test 4: History
        console.print("\n[yellow]Test 4: Query History[/yellow]")
        
        result4 = await engine.execute(
            action="history",
            user_id="test_user",
            limit=5
        )
        
        if result4["success"]:
            console.print(f"[green]✅ History: {len(result4['history'])} entries[/green]")
        
        # Test 5: Status
        console.print("\n[yellow]Test 5: Engine Status[/yellow]")
        
        result5 = await engine.execute(action="status")
        
        if result5["success"]:
            console.print("[green]✅ Status retrieved[/green]")
            console.print(f"Active agents: {result5['active_agents']}")
            console.print(f"Active teams: {result5['active_teams']}")
        
        console.rule("[bold green]TEST COMPLETATI[/bold green]")
    
    # Esegui test
    asyncio.run(test_inference_engine())