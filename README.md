# 🤖 AI Agent Inference Module

[![Streamlit](https://img.shields.io/badge/Streamlit-1.47.1-FF4B4B.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Una **interfaccia Streamlit professionale** per la gestione di agenti AI, esecuzione di inferenze e gestione di sistemi di memoria e conoscenza. L'applicazione supporta più provider di modelli (Ollama, OpenAI, Anthropic, Gemini) e permette l'utilizzo simultaneo da parte di massimo 5 utenti.

![Login Interface](https://github.com/user-attachments/assets/3d2e67be-5691-4ebb-a9d1-0370e26f1678)

## ✨ Caratteristiche Principali

### 🔐 Sistema Multi-Utente
- **Autenticazione sicura** con registrazione e login
- **Supporto per massimo 5 utenti simultanei**
- **Gestione sessioni** con timeout automatico
- **Interfaccia utente personalizzata** per ogni utente

### 🤖 Gestione Agenti Avanzata
- **Creazione agenti** con configurazione completa
- **Supporto per tutti i principali provider**:
  - 🦙 **Ollama** (con scan automatico modelli locali)
  - 🤖 **OpenAI** (GPT-4o, GPT-4 Turbo, GPT-3.5)
  - 🧠 **Anthropic** (Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku)
  - 💎 **Google Gemini** (Gemini 1.5 Pro/Flash, Gemini Pro)
- **System prompts** personalizzabili (hardcoded e custom salvabili)
- **Assegnazione tools** agli agenti
- **Integrazione knowledge base**

![Dashboard](https://github.com/user-attachments/assets/317784c5-ad32-4751-a9be-3498b4923202)

### 🧠 Interfaccia di Inferenza
- **Chat interface in tempo reale** con streaming
- **Batch processing** per query multiple
- **Storico inferenze** con export
- **Visualizzazione tool calls e reasoning**
- **Metriche performance** dettagliate

### ⚙️ Configurazione Modelli
- **Scan automatico modelli Ollama** presenti in locale
- **Installazione guidata** modelli popolari
- **Test connessione** per tutti i provider
- **Gestione API keys** sicura
- **Configurazione avanzata** per ogni provider

![Settings - Ollama Configuration](https://github.com/user-attachments/assets/b894d263-6cad-4767-9864-ad1d6f0ef5cc)

### 💾 Sistema Memoria e Conoscenza
- **Modulo memoria** per conversazioni persistenti
- **Knowledge base** con gestione documenti
- **Chunking intelligente** dei documenti
- **Ricerca semantica** (implementazione base)
- **Assegnazione knowledge** agli agenti

### 🎨 Interfaccia Professionale
- **Design moderno** con gradiente personalizzato
- **Layout responsive** ottimizzato per uso desktop
- **Navigazione intuitiva** con sidebar organizzata
- **Feedback visuale** per tutte le operazioni
- **Tema professionale** con colori personalizzati

![Agent Management](https://github.com/user-attachments/assets/3d485f23-2fc9-48f2-a378-8b31f349c587)

## 🚀 Installazione e Avvio

### Prerequisiti
```bash
Python 3.8+
pip (package manager)
```

### 1. Clona il Repository
```bash
git clone https://github.com/michele1967lux/streamlit-inference-module.git
cd streamlit-inference-module
```

### 2. Installa le Dipendenze
```bash
pip install -r requirements.txt
```

### 3. Avvia l'Applicazione

#### Metodo 1: Script di Avvio
```bash
python run.py
```

#### Metodo 2: Streamlit Diretto
```bash
streamlit run app.py --server.port=8501
```

### 4. Accedi all'Interfaccia
Apri il browser su: **http://localhost:8501**

## 📋 Utilizzo dell'Applicazione

### 1. Registrazione e Login
- Crea un nuovo account nella schermata di login
- Effettua il login con le credenziali create
- L'interfaccia mostrerà il dashboard principale

### 2. Configurazione Modelli

#### Ollama (Locale)
1. Vai su **Settings → Model Configuration → Ollama**
2. Verifica l'URL di Ollama (default: `http://localhost:11434`)
3. Clicca **"Test Connection"** per verificare la connessione
4. Clicca **"Scan Models"** per rilevare i modelli installati
5. Installa nuovi modelli dalla sezione **"Popular Models"**

#### OpenAI/Anthropic/Gemini
1. Vai alla tab del provider desiderato
2. Inserisci la tua **API Key**
3. Seleziona il **modello predefinito**
4. Salva la configurazione

### 3. Creazione Agenti
1. Vai su **Agent Management → Create Agent**
2. Compila le informazioni base (**Nome**, **Descrizione**)
3. Seleziona il **provider** e **modello**
4. Configura il **system prompt** (predefinito o custom)
5. Imposta parametri avanzati (temperatura, max tokens)
6. Crea l'agente

### 4. Assegnazione Tools e Knowledge
1. Vai su **Agent Management → Tools & Knowledge**
2. Seleziona l'agente da configurare
3. Assegna **tools** disponibili (Web Search, Calculator, ecc.)
4. Assegna **knowledge bases** create

### 5. Esecuzione Inferenze
1. Vai su **Inference Interface → Chat Interface**
2. Seleziona l'agente da utilizzare
3. Configura le opzioni (streaming, reasoning, tool calls)
4. Scrivi la query e clicca **"Send Message"**
5. Visualizza la risposta in tempo reale

### 6. Batch Processing
1. Vai su **Inference Interface → Batch Processing**
2. Inserisci multiple query (manualmente o via file)
3. Configura le opzioni di batch
4. Esegui il batch e scarica i risultati

## 🔧 Configurazione Avanzata

### System Prompts Personalizzati
```python
# Esempi di system prompts disponibili
prompts = {
    "default": "You are a helpful AI assistant...",
    "creative": "You are a creative AI assistant...",
    "analytical": "You are an analytical AI assistant...",
    "custom": "Il tuo prompt personalizzato..."
}
```

### Configurazione Ollama
```bash
# Installazione modelli consigliati
ollama pull llama3.1:8b      # Modello generale veloce
ollama pull mistral:7b       # Alternativa efficiente
ollama pull codellama:7b     # Specializzato per codice
ollama pull phi3:mini        # Compatto e veloce
```

### Tools Disponibili
- 🔍 **Web Search**: Ricerca informazioni sul web
- 🧮 **Calculator**: Calcoli matematici
- 📄 **File Reader**: Lettura e analisi file
- 💻 **Code Executor**: Esecuzione codice Python
- 🧠 **Reasoning**: Tools di ragionamento avanzato
- 🧠 **Memory Search**: Ricerca nella memoria conversazioni

## 📁 Struttura del Progetto

```
streamlit-inference-module/
├── app.py                          # Applicazione principale Streamlit
├── run.py                          # Script di avvio
├── requirements.txt                # Dipendenze Python
├── README.md                       # Documentazione
├── 
├── utils/                          # Moduli utility
│   ├── config_manager.py           # Gestione configurazione
│   ├── user_manager.py             # Gestione utenti
│   ├── model_scanner.py            # Scanner modelli
│   └── knowledge_manager.py        # Gestione knowledge base
├── 
├── components/                     # Componenti UI
│   ├── agent_manager.py            # Gestione agenti
│   ├── inference_interface.py      # Interfaccia inferenza
│   └── settings_manager.py         # Gestione settings
├── 
├── inference_engine_module/        # Modulo inferenza esistente
│   ├── inference_engine_module.py  # Engine inferenza completo
│   └── inference_engine_module_ori.py
├── 
├── memory_agent_module/            # Modulo memoria esistente
│   └── memory_agent_module.py      # Gestione memoria agenti
└── 
└── data/                           # Directory dati (auto-creata)
    ├── config.json                 # Configurazione app
    ├── users.db                    # Database utenti
    ├── knowledge.db                # Database knowledge
    └── inference/                  # Storage risultati inferenza
```

## 🔒 Sicurezza e Limitazioni

### Sicurezza
- **Password hashing** con PBKDF2 e salt
- **Session management** sicuro
- **Validazione input** per prevenire injection
- **API keys** memorizzate in modo sicuro
- **Timeout sessioni** automatico (24 ore)

### Limitazioni
- **Massimo 5 utenti** simultanei
- **Dipendenza da servizi esterni** (OpenAI, Anthropic, Gemini)
- **Ollama richiede installazione locale**
- **Knowledge base** usa ricerca testuale semplice (no embeddings)

## 🛠️ Sviluppo e Personalizzazione

### Aggiungere Nuovi Provider
1. Estendi `ModelScanner` in `utils/model_scanner.py`
2. Aggiungi configurazione in `SettingsManagerComponent`
3. Aggiorna `AgentManagerComponent` per supporto provider

### Personalizzare l'UI
1. Modifica CSS in `app.py` → `_apply_custom_css()`
2. Aggiorna colori e stili nel tema
3. Personalizza layout nei componenti

### Aggiungere Nuovi Tools
1. Definisci tool in `AgentManagerComponent._get_available_tools()`
2. Implementa logica tool nell'inference engine
3. Aggiorna UI per configurazione tool

## 📊 Monitoring e Logs

### Logs Applicazione
```bash
# I logs sono salvati in:
data/inference/inference.log     # Logs inferenza
data/users.db                    # Gestione utenti
data/config.json                 # Configurazione
```

### Metriche Disponibili
- **Utenti attivi** e registrazioni
- **Agenti creati** e configurati
- **Inferenze eseguite** con timing
- **Utilizzo modelli** per provider
- **Performance** sistema

## 🤝 Contribuzione

### Come Contribuire
1. Fork del repository
2. Crea feature branch (`git checkout -b feature/amazing-feature`)
3. Commit modifiche (`git commit -m 'Add amazing feature'`)
4. Push al branch (`git push origin feature/amazing-feature`)
5. Apri Pull Request

### Aree di Miglioramento
- [ ] **Embeddings vettoriali** per knowledge base
- [ ] **Plugin system** per tools personalizzati
- [ ] **Multi-tenancy** avanzata
- [ ] **Caching** risposta AI
- [ ] **Analytics** uso avanzate
- [ ] **API REST** per integrazioni
- [ ] **Docker deployment**
- [ ] **Autenticazione OAuth**

## 📄 Licenza

Questo progetto è rilasciato sotto licenza **MIT**. Vedi [LICENSE](LICENSE) per dettagli.

## 🆘 Supporto

### Problemi Comuni

**Q: Ollama non si connette**
A: Verifica che Ollama sia avviato (`ollama serve`) e accessibile su `localhost:11434`

**Q: Errore "No module named 'agno'"**
A: Questo è normale - il modulo AGNO è opzionale. L'app funziona in modalità demo.

**Q: API Key non funziona**
A: Verifica validità API key e crediti disponibili per il provider.

**Q: Slow performance**
A: Riduci concurrent users o usa modelli più piccoli per Ollama.

### Contatti
- **Repository**: [GitHub](https://github.com/michele1967lux/streamlit-inference-module)
- **Issues**: [GitHub Issues](https://github.com/michele1967lux/streamlit-inference-module/issues)

---

**Sviluppato con ❤️ usando Streamlit e Python**