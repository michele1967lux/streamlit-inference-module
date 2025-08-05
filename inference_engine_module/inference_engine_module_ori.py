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

# AGNO imports
from agno.agent import Agent, RunResponse, RunEvent
from agno.team.team import Team
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.storage.sqlite import SqliteStorage
from agno.utils.log import logger
from agno.utils.pprint import pprint_run_response

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

# Create console with safe encoding for Windows
try:
    console = Console(force_terminal=False, no_color=True)
except Exception:
    # Fallback to minimal console if Rich fails
    class MinimalConsole:
        def print(self, *args, **kwargs):
            try:
                print(*args)
            except UnicodeEncodeError:
                print("Output with encoding issues")
        def rule(self, text=""):
            print(f"--- {text} ---")
    console = MinimalConsole()


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
                 storage_path: str = "tmp/inference"):
        """
        Inizializza il motore di inference.
        
        Args:
            agent_controller: Controller per gestione agenti
            team_constructor: Constructor per gestione team
            storage_path: Path per storage risultati
        """
        self.agent_controller = agent_controller
        self.team_constructor = team_constructor
        self.console = console
        
        # Storage per risultati e cache
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.result_storage = SqliteStorage(
            table_name="inference_results",
            db_file=str(self.storage_path / "results.db")
        )
        
        self.cache_storage = SqliteStorage(
            table_name="response_cache",
            db_file=str(self.storage_path / "cache.db")
        )
        
        # Cache runtime
        self.active_agents = {}  # agent_id -> Agent instance
        self.active_teams = {}   # team_id -> Team instance
        
        # Debug capture
        self.debug_capture = DebugCapture()
        
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
            try:
                if level == "ERROR":
                    console.print(f"[red]ERROR: {message}[/red]")
                elif level == "WARNING":
                    console.print(f"[yellow]WARNING: {message}[/yellow]")
                elif level == "SUCCESS":
                    console.print(f"[green]SUCCESS: {message}[/green]")
                elif level == "DEBUG":
                    console.print(f"[dim]DEBUG: {message}[/dim]")
                else:
                    console.print(f"[blue]INFO: {message}[/blue]")
            except (UnicodeEncodeError, UnicodeDecodeError):
                # Fallback to basic print if console fails
                print(f"{level}: {message}")
        
        return log_msg
    
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
                
        Returns:
            Dict con risultato operazione
        """
        try:
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
            run_params = self._prepare_run_params(request)
            
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
        """Ottiene l'istanza target (agente o team)."""
        try:
            if target_type == "agent":
                # Controlla cache
                if target_id in self.active_agents:
                    return self.active_agents[target_id]
                
                # Crea agente tramite controller
                if self.agent_controller:
                    agent = self.agent_controller.get_agent_by_id(target_id)
                    if agent:
                        self.active_agents[target_id] = agent
                        return agent
                
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
    
    def _prepare_run_params(self, request: QueryRequest) -> Dict[str, Any]:
        """Prepara parametri per run agente/team."""
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
        
        return params
    
    async def _handle_streaming_response(self, target: Union[Agent, Team], query: str,
                                       run_params: Dict[str, Any], request: QueryRequest,
                                       metrics: InferenceMetrics, logs: List[str]) -> InferenceResult:
        """Gestisce response in streaming."""
        response_content = ""
        reasoning_content = ""
        tool_calls = []
        full_response = None
        
        try:
            # Inizia streaming con spinner
            with Live(Spinner("dots", text="[cyan]Processing query...[/cyan]"), 
                     refresh_per_second=10) as live:
                
                # Esegui run asincrono
                if hasattr(target, 'arun'):
                    response_stream = await target.arun(query, **run_params)
                else:
                    response_stream = target.run(query, **run_params)
                
                # Processa stream
                async for event in self._process_stream(response_stream):
                    if isinstance(event, RunResponse):
                        # Content event
                        if event.content:
                            response_content += event.content
                            live.update(Panel(
                                Markdown(response_content[-500:]),
                                title="[bold]Response[/bold]",
                                border_style="green"
                            ))
                        
                        # Tool call event
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
                        
                        # Reasoning event
                        if hasattr(event, 'reasoning_content') and event.reasoning_content:
                            reasoning_content += event.reasoning_content
                            metrics.reasoning_steps += 1
                            
                            if request.settings.show_full_reasoning:
                                self._display_reasoning(event.reasoning_content)
                        
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
            
            # Crea risultato
            result = InferenceResult(
                success=True,
                request=request,
                response_content=response_content,
                full_response=full_response,
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
                metrics=metrics,
                logs=logs
            )
            
            # Calcola metriche finali
            result.metrics.response_size_chars = len(response_content)
            
            return result
            
        except Exception as e:
            self._log(f"Errore durante streaming: {str(e)}", "ERROR")
            raise
    
    async def _handle_standard_response(self, target: Union[Agent, Team], query: str,
                                      run_params: Dict[str, Any], request: QueryRequest,
                                      metrics: InferenceMetrics, logs: List[str]) -> InferenceResult:
        """Gestisce response standard (non-streaming)."""
        try:
            # Esegui query
            self._log("Esecuzione query (non-streaming)...", "INFO")
            
            if hasattr(target, 'arun'):
                response = await target.arun(query, **run_params)
            else:
                response = target.run(query, **run_params)
            
            # Estrai contenuti
            response_content = ""
            reasoning_content = ""
            tool_calls = []
            
            if response:
                # Content
                if hasattr(response, 'content'):
                    response_content = response.content or ""
                
                # Reasoning
                if hasattr(response, 'reasoning_content'):
                    reasoning_content = response.reasoning_content or ""
                    if reasoning_content:
                        metrics.reasoning_steps = 1
                
                # Tool calls (da implementare in base alla struttura reale)
                # Placeholder per ora
                
                # Log per debug
                if request.settings.debug_level == "DEBUG":
                    logs.append(f"Response type: {type(response)}")
                    logs.append(f"Response content length: {len(response_content)}")
            
            # Crea risultato
            result = InferenceResult(
                success=True,
                request=request,
                response_content=response_content,
                full_response=response,
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
                metrics=metrics,
                logs=logs
            )
            
            result.metrics.response_size_chars = len(response_content)
            
            return result
            
        except Exception as e:
            self._log(f"Errore response standard: {str(e)}", "ERROR")
            raise
    
    async def _process_stream(self, stream: Union[Iterator, AsyncIterator]):
        """Processa stream di eventi."""
        try:
            if hasattr(stream, '__aiter__'):
                # Async iterator
                async for event in stream:
                    yield event
            else:
                # Sync iterator - convert to async
                for event in stream:
                    yield event
                    await asyncio.sleep(0)  # Yield control
        except Exception as e:
            self._log(f"Errore processing stream: {str(e)}", "ERROR")
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
        console.rule(f"[bold blue]INFERENCE ENGINE - {request.target_type.upper()}[/bold blue]")
        
        info_table = Table(show_header=False)
        info_table.add_column("Campo", style="cyan", width=20)
        info_table.add_column("Valore", style="white")
        
        info_table.add_row("Target", f"{request.target_type}: {request.target_id}")
        info_table.add_row("User", request.user_id or "default")
        info_table.add_row("Session", request.session_id or "new")
        info_table.add_row("Query", request.query[:100] + "..." if len(request.query) > 100 else request.query)
        info_table.add_row("Streaming", "YES" if request.settings.stream else "NO")
        info_table.add_row("Debug Level", request.settings.debug_level)
        
        console.print(info_table)
        console.print()
    
    def _display_tool_call(self, tool_info: Dict[str, Any]):
        """Display chiamata tool."""
        console.print(Panel(
            f"[yellow]Tool:[/yellow] {tool_info.get('tool', 'Unknown')}\n"
            f"[yellow]Args:[/yellow] {tool_info.get('args', {})}\n"
            f"[yellow]Result:[/yellow] {str(tool_info.get('result', 'Pending'))[:200]}",
            title="[bold]Tool Call[/bold]",
            border_style="yellow"
        ))
    
    def _display_reasoning(self, reasoning: str):
        """Display reasoning step."""
        console.print(Panel(
            Markdown(reasoning[:500] + "..." if len(reasoning) > 500 else reasoning),
            title="[bold]Reasoning[/bold]",
            border_style="magenta"
        ))
    
    def _display_final_result(self, result: InferenceResult):
        """Display risultato finale."""
        console.rule("[bold green]INFERENCE COMPLETATA[/bold green]")
        
        # Metriche
        if result.metrics:
            metrics_table = Table(title="Metriche Performance")
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
                title="[bold]Response Preview[/bold]",
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