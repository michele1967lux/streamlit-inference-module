# Inference Engine Module

## Descrizione
Modulo avanzato per l'esecuzione di query su agenti e team AGNO con logging completo, metriche performance e configurazioni personalizzabili.

## Caratteristiche Principali
- ✅ Esecuzione query su agenti singoli e team
- ✅ Logging dettagliato con livelli configurabili  
- ✅ Streaming responses con tracking intermedio
- ✅ Gestione reasoning e tool calls
- ✅ Cache risultati per performance
- ✅ Export risultati in formati multipli
- ✅ Metriche performance e token usage
- ✅ Support per modalità batch

## Classi Principali

### 1. InferenceSettings (Dataclass)
Configurazioni avanzate per l'inference:
```python
@dataclass
class InferenceSettings:
    stream: bool = True                        # Streaming delle risposte
    show_tool_calls: bool = True               # Mostra chiamate tool
    show_full_reasoning: bool = False          # Reasoning completo
    stream_intermediate_steps: bool = True     # Steps intermedi
    markdown: bool = True                      # Output markdown
    save_to_file: Optional[str] = None         # Salva su file
    cache_responses: bool = True               # Cache risposte
    timeout_seconds: Optional[int] = None      # Timeout operazioni
    max_retries: int = 3                       # Retry automatici
    temperature: Optional[float] = None        # Temperature modello
    max_tokens: Optional[int] = None           # Max tokens
    debug_level: str = "INFO"                  # Livello debug
    export_format: str = "json"                # Formato export
    track_metrics: bool = True                 # Tracking metriche
```

### 2. QueryRequest (Dataclass)
Richiesta strutturata:
```python
@dataclass
class QueryRequest:
    query: str                                 # Testo query
    target_type: str                           # "agent" o "team"
    target_id: str                             # ID target
    settings: InferenceSettings               # Configurazioni
    context: Optional[Dict[str, Any]] = None  # Contesto aggiuntivo
    session_id: Optional[str] = None          # ID sessione
    user_id: Optional[str] = None             # ID utente
```

### 3. InferenceMetrics (Dataclass)
Metriche performance:
```python
@dataclass
class InferenceMetrics:
    start_time: float                         # Timestamp inizio
    end_time: Optional[float] = None          # Timestamp fine
    duration_seconds: Optional[float] = None  # Durata totale
    tokens_used: Optional[int] = None         # Token utilizzati
    tool_calls_count: int = 0                 # Numero tool calls
    reasoning_steps: int = 0                  # Steps reasoning
    error_count: int = 0                      # Errori
    retry_count: int = 0                      # Retry
    response_size_chars: int = 0              # Dimensione risposta
```

### 4. InferenceResult (Dataclass)
Risultato completo:
```python
@dataclass
class InferenceResult:
    success: bool                             # Successo operazione
    request: QueryRequest                     # Richiesta originale
    response_content: Optional[str] = None    # Contenuto risposta
    full_response: Optional[RunResponse] = None # Risposta completa
    reasoning_content: Optional[str] = None   # Contenuto reasoning
    tool_calls: List[Dict[str, Any]]         # Lista tool calls
    metrics: Optional[InferenceMetrics] = None # Metriche
    logs: List[str]                          # Log operazioni
    error: Optional[str] = None              # Errore se presente
    timestamp: str                           # Timestamp ISO
```

### 5. InferenceEngineModule
Classe principale del motore:

#### Metodi Principali:
```python
async def execute(self, request: QueryRequest) -> InferenceResult
async def execute_batch(self, requests: List[QueryRequest]) -> List[InferenceResult]
async def execute_streaming(self, request: QueryRequest) -> AsyncIterator[InferenceResult]
def get_cached_result(self, cache_key: str) -> Optional[InferenceResult]
def save_result_to_file(self, result: InferenceResult, format: str = "json")
def get_metrics_summary(self, results: List[InferenceResult]) -> Dict[str, Any]
def clear_cache(self)
def export_session_data(self, session_id: str, format: str = "json") -> str
```

## API di Utilizzo

### 1. Esecuzione Query Singola
```python
engine = InferenceEngineModule(
    agents_registry=agents_dict,
    teams_registry=teams_dict
)

# Configurazioni personalizzate
settings = InferenceSettings(
    stream=True,
    show_tool_calls=True,
    debug_level="DEBUG",
    export_format="markdown"
)

# Richiesta
request = QueryRequest(
    query="Analizza NVDA vs TSLA",
    target_type="agent",
    target_id="finance_agent",
    settings=settings,
    session_id="session_123"
)

# Esecuzione
result = await engine.execute(request)

if result.success:
    print(result.response_content)
    print(f"Durata: {result.metrics.duration_seconds}s")
    print(f"Token usati: {result.metrics.tokens_used}")
```

### 2. Esecuzione Batch
```python
requests = [
    QueryRequest("Query 1", "agent", "agent1", settings),
    QueryRequest("Query 2", "agent", "agent2", settings),
    QueryRequest("Query 3", "team", "team1", settings)
]

results = await engine.execute_batch(requests)

for i, result in enumerate(results):
    print(f"Risultato {i+1}: {result.success}")
```

### 3. Streaming con Live Updates
```python
async for partial_result in engine.execute_streaming(request):
    if partial_result.response_content:
        print(f"Aggiornamento: {partial_result.response_content}")
```

## Integrazione in UI

### 1. Form Configurazione
```python
# Sezione Impostazioni Inference
{
    "streaming": bool,              # Checkbox streaming
    "show_tools": bool,             # Checkbox tool calls
    "debug_level": str,             # Dropdown: DEBUG, INFO, WARNING, ERROR
    "export_format": str,           # Radio: json, markdown, txt
    "timeout": int,                 # Slider timeout (sec)
    "max_retries": int,             # Number input retry
    "temperature": float,           # Slider temperature 0.0-2.0
    "max_tokens": int,              # Number input tokens
    "cache_enabled": bool,          # Checkbox cache
    "track_metrics": bool           # Checkbox metriche
}
```

### 2. Dashboard Esecuzione
```python
# Progress tracking in tempo reale
{
    "status": "running|completed|error",
    "progress_bar": float,          # 0.0-1.0
    "current_step": str,            # "Executing query...", "Processing tools..."
    "elapsed_time": float,          # Secondi trascorsi
    "estimated_remaining": float,   # Stima tempo rimanente
    "tokens_used": int,             # Token utilizzati finora
    "tool_calls": int               # Tool calls eseguiti
}
```

### 3. Risultati Visualizzazione
```python
# Pannello risultati
{
    "response_content": str,        # Contenuto principale (markdown)
    "reasoning_steps": List[str],   # Steps reasoning se abilitato
    "tool_calls_log": List[Dict],   # Log tool calls dettagliato
    "metrics_summary": Dict,        # Riepilogo metriche
    "export_options": List[str],    # Formati export disponibili
    "cache_status": str,            # "hit|miss|disabled"
    "error_details": Optional[str]  # Dettagli errore se presente
}
```

### 4. Sessione Management
```python
# Gestione sessioni
{
    "session_id": str,              # ID sessione corrente
    "query_history": List[Dict],    # Storico query sessione
    "total_tokens": int,            # Token totali sessione
    "total_duration": float,        # Durata totale
    "success_rate": float,          # % successo query
    "export_session": callable,     # Export completo sessione
    "clear_session": callable       # Reset sessione
}
```

## Pattern di Estensione

### 1. Custom Metrics Collector
```python
class CustomMetricsCollector:
    def collect_custom_metrics(self, result: InferenceResult) -> Dict[str, Any]:
        # Logica custom per metriche aggiuntive
        return {"custom_metric": value}
```

### 2. Result Processor
```python
class ResultProcessor:
    def process_result(self, result: InferenceResult) -> InferenceResult:
        # Post-processing del risultato
        return modified_result
```

### 3. Cache Strategy
```python
class CustomCacheStrategy:
    def get_cache_key(self, request: QueryRequest) -> str:
        # Logica custom per chiavi cache
        return custom_key
```

## Logging e Debug

### Livelli di Debug
- **DEBUG**: Tutti i dettagli interni
- **INFO**: Informazioni principali operazioni
- **WARNING**: Avvertimenti e situazioni anomale
- **ERROR**: Solo errori critici

### Output Formattato
Il modulo utilizza Rich per output colorato e formattato:
- **Tabelle**: Metriche performance
- **Panel**: Risultati principali
- **Progress**: Barre progresso
- **Syntax**: Highlighting codice
- **Markdown**: Rendering risultati

## File Export

### Formati Supportati
- **JSON**: Struttura completa con metadati
- **Markdown**: Formato human-readable
- **TXT**: Testo semplice

### Esempio Export JSON
```json
{
    "success": true,
    "request": {
        "query": "Analizza NVDA",
        "target_type": "agent",
        "target_id": "finance_agent"
    },
    "response_content": "...",
    "metrics": {
        "duration_seconds": 3.45,
        "tokens_used": 1250,
        "tool_calls_count": 2
    },
    "timestamp": "2024-01-15T10:30:45"
}
```

## Test e Demo

### Test Automatici
```python
if __name__ == "__main__":
    asyncio.run(test_inference_engine())
```

Il modulo include test completi per:
- ✅ Esecuzione query singole
- ✅ Batch processing
- ✅ Streaming responses
- ✅ Cache management
- ✅ Export functionality
- ✅ Error handling

## Dipendenze
```txt
agno[all]
rich>=13.0.0
asyncio (built-in)
dataclasses (built-in Python 3.7+)
json (built-in)
logging (built-in)
pathlib (built-in)
```

## Test Status
❌ **Unicode Error**: Errore encoding console Windows
✅ **Architettura**: Design modulare e robusto
✅ **Async Support**: Supporto completo asyncio
✅ **Rich Integration**: Output formattato avanzato
✅ **Metrics System**: Sistema metriche completo
✅ **Cache System**: Cache implementata
✅ **Export System**: Export multi-formato