# adk-go-memory

Production-grade memory layer for Google ADK-Go agents with automatic observation extraction, hybrid search (vector + FTS), and intelligent context assembly.

## Overview

`adk-go-memory` provides a complete memory management system for Google ADK-Go agents. It extracts facts from conversations, stores them in SQLite with vector and full-text search capabilities, and retrieves relevant context to augment agent responses.

### Key Features

- **Automatic Observation Extraction**: LLM-powered fact extraction from conversations with confidence levels (explicit, deductive, inductive, contradiction)
- **Hybrid Search**: Combines sqlite-vec vector similarity search with FTS5 full-text search via Reciprocal Rank Fusion (RRF)
- **Deduplication**: Automatically detects and merges near-duplicate observations
- **Working Representations**: Multi-strategy context assembly combining semantic search, most-derived facts, and recent observations
- **Peer Cards**: User profiles with up to 40 scored facts for persistent user modeling
- **ADK-Go Integration**: Implements `google.golang.org/adk/memory.Service` interface
- **Production Ready**: SQLite-based storage with CGO-enabled sqlite-vec for efficient vector operations

## Installation

```bash
go get github.com/ieshan/adk-go-memory
```

### Requirements

- Go 1.26+
- CGO enabled (for sqlite-vec)
- SQLite with FTS5 support

### Build Tags

The package requires the `sqlite_fts5` build tag:

```bash
CGO_ENABLED=1 go build -tags=sqlite_fts5 ./...
```

## Quick Start

### Basic Usage

```go
package main

import (
    "context"
    "log"

    memory "github.com/ieshan/adk-go-memory"
    "github.com/ieshan/adk-go-memory/adapter"
    adkmemory "google.golang.org/adk/memory" // ADK interface
)

func main() {
    ctx := context.Background()

    // Create in-memory storage (use NewSQLiteStorage for persistence)
    storage, err := adapter.InMemory()
    if err != nil {
        log.Fatal(err)
    }
    defer storage.Close()

    // Create a memory service
    svc := memory.NewService(memory.ServiceConfig{
        Storage: storage,
    })
    defer svc.Close()

    // Service implements google.golang.org/adk/memory.Service
    var _ adkmemory.Service = svc
}
```

## API Reference

### Service

The `Service` is the main entry point that implements `google.golang.org/adk/memory.Service`.

```go
// ServiceConfig configures the Service.
type ServiceConfig struct {
    Storage  adapter.Storage  // Storage backend for observations
    Deriver  *Deriver           // Optional: LLM-powered fact extraction
    Provider *Provider          // Optional: Context assembly provider
}

// NewService creates a new Service with the given configuration.
func NewService(cfg ServiceConfig) *Service

// AddSessionToMemory extracts observations from session events and stores them.
// Requires a Deriver to be configured.
func (s *Service) AddSessionToMemory(ctx context.Context, sess session.Session) error

// SearchMemory searches the memory for relevant observations.
func (s *Service) SearchMemory(ctx context.Context, req *memory.SearchRequest) (*memory.SearchResponse, error)

// SetDeriver sets the deriver on the service (useful for lazy initialization).
func (s *Service) SetDeriver(d *Deriver)

// SetProvider sets the provider on the service.
func (s *Service) SetProvider(p *Provider)

// Close releases resources held by the service.
func (s *Service) Close() error
```

### Deriver

The `Deriver` uses an LLM to automatically extract factual observations from conversations.

```go
// DeriverConfig configures the Deriver.
type DeriverConfig struct {
    LLM     model.LLM      // LLM for observation extraction
    Storage adapter.Storage // Storage for saving observations
}

// NewDeriver creates a new Deriver with the given configuration.
func NewDeriver(cfg DeriverConfig) *Deriver

// Derive extracts observations from timestamped messages and stores them.
// Automatically deduplicates near-duplicate observations.
func (d *Deriver) Derive(ctx context.Context, messages []TimestampedMessage, sessionID string) error
```

### Provider

The `Provider` wires together all memory components for comprehensive context assembly.

```go
// ProviderConfig configures the Provider.
type ProviderConfig struct {
    Storage       adapter.Storage                        // Storage backend
    Deriver       *Deriver                               // Optional: Fact extraction
    Summarizer    *Summarizer                            // Optional: Periodic summaries
    Dialectic     *Dialectic                             // Optional: LLM-powered Q&A
    EmbeddingFunc func(ctx context.Context, text string) ([]float32, error) // For semantic search
}

// NewProvider creates a new Provider with the given configuration.
func NewProvider(cfg ProviderConfig) *Provider

// GetMemoryContext retrieves memory context for the given query using the representation manager.
func (p *Provider) GetMemoryContext(ctx context.Context, query string, sessionID, userID, appName string) (string, error)

// GetOrCreatePeerCard gets or creates a peer card for the given peer ID.
func (p *Provider) GetOrCreatePeerCard(peerID string) *PeerCard

// LoadPeerCardFromMemory loads peer card facts from storage.
func (p *Provider) LoadPeerCardFromMemory(ctx context.Context, peerID string) error

// OnSessionStart is called when a session starts to load peer cards.
func (p *Provider) OnSessionStart(ctx context.Context, sessionID, userID, appName string) error

// Close releases resources held by the provider.
func (p *Provider) Close() error
```

### RepresentationManager

The `RepresentationManager` assembles working representations using multiple retrieval strategies.

```go
// RepresentationConfig configures the RepresentationManager.
type RepresentationConfig struct {
    Storage           adapter.Storage
    SemanticBudget    int                                    // Max semantic search results
    MostDerivedBudget int                                    // Max most-derived observations
    RecentBudget      int                                    // Max recent observations
    EmbeddingFunc     func(ctx context.Context, text string) ([]float32, error)
}

// NewRepresentationManager creates a new RepresentationManager.
func NewRepresentationManager(cfg RepresentationConfig) *RepresentationManager

// GetWorkingRepresentation retrieves a working representation combining semantic,
// most-derived, and recent observations.
func (rm *RepresentationManager) GetWorkingRepresentation(ctx context.Context, query, sessionID, userID, appName string) (*WorkingRepresentation, error)

// WorkingRepresentation holds assembled observations for context injection.
type WorkingRepresentation struct {
    Observations  []adapter.Observation
    SemanticCount int
    DerivedCount  int
    RecentCount   int
    TotalScore    float64
}

// Format returns a formatted string with [semantic], [derived], [recent] prefixes.
func (wr *WorkingRepresentation) Format() string
```

### PeerCard

`PeerCard` stores scored facts about users (peers) with a capacity of 40 facts.

```go
// PeerFact represents a single fact about a peer.
type PeerFact struct {
    Content string                   // The fact content
    Score   float64                  // Confidence score (0.0 - 1.0)
    Type    adapter.ObservationLevel  // explicit, deductive, inductive, contradiction
    Tags    []string                 // Categorical tags
}

// PeerCard stores facts about a peer (user).
type PeerCard struct {
    // peerID and facts are internal
}

// NewPeerCard creates a new peer card for the given peer ID.
func NewPeerCard(peerID string) *PeerCard

// PeerID returns the ID of the peer this card represents.
func (pc *PeerCard) PeerID() string

// AddFact adds a fact to the peer card. Evicts lowest-scoring fact if at capacity.
func (pc *PeerCard) AddFact(fact PeerFact)

// Facts returns all facts in the peer card.
func (pc *PeerCard) Facts() []PeerFact

// ReplaceFacts replaces all facts with the given slice.
func (pc *PeerCard) ReplaceFacts(facts []PeerFact)

// Prune removes facts with scores below the threshold.
func (pc *PeerCard) Prune(threshold float64)

// ByType returns facts filtered by observation level.
func (pc *PeerCard) ByType(level adapter.ObservationLevel) []PeerFact

// ByTag returns facts that have the given tag.
func (pc *PeerCard) ByTag(tag string) []PeerFact

// Render returns a formatted string representation of the peer card.
func (pc *PeerCard) Render() string
```

### Summarizer

The `Summarizer` creates periodic conversation summaries at defined intervals.

```go
// SummarizerConfig configures the summarizer.
type SummarizerConfig struct {
    LLM           model.LLM
    Storage       adapter.Storage
    ShortInterval int // Messages per short summary (default: 20)
    LongInterval  int // Messages per long summary (default: 60)
}

// NewSummarizer creates a new summarizer.
func NewSummarizer(cfg SummarizerConfig) *Summarizer

// ShouldSummarize checks if a summary should be created at this message count.
// Returns the summary type and true if a summary should be created.
func (s *Summarizer) ShouldSummarize(messageCount int) (SummaryType, bool)

// Summarize creates a summary of the given messages.
func (s *Summarizer) Summarize(ctx context.Context, messages []TimestampedMessage) (*Summary, error)

// StoreSummary persists a summary to storage as a special observation.
func (s *Summarizer) StoreSummary(ctx context.Context, sessionID string, summary *Summary) error

// GetBothSummaries retrieves short and long summaries for a session.
func (s *Summarizer) GetBothSummaries(ctx context.Context, sessionID string) (*Summary, *Summary, error)
```

### Dialectic

The `Dialectic` provides LLM-powered memory query capabilities with synthesized answers.

```go
// DialecticConfig configures the Dialectic.
type DialecticConfig struct {
    LLM        model.LLM
    Storage    adapter.Storage
    MaxResults int // Default: 10
}

// NewDialectic creates a new Dialectic with the given configuration.
func NewDialectic(cfg DialecticConfig) *Dialectic

// Query performs a memory-based query and returns a synthesized answer.
func (d *Dialectic) Query(ctx context.Context, query string, opts QueryOptions) (string, error)
```

### MemoryTool

`MemoryTool` implements the ADK tool interface, allowing agents to search their own memory.

```go
// SearchMemoryArgs matches the tool schema expected by the LLM.
type SearchMemoryArgs struct {
    Query      string `json:"query"`
    MaxResults int    `json:"max_results,omitempty"`
}

// ToolObservation represents a single observation in tool results.
type ToolObservation struct {
    Content string   `json:"content"`
    Level   string   `json:"level"`
    Tags    []string `json:"tags,omitempty"`
}

// NewMemoryTool creates a new memory search tool.
func NewMemoryTool(storage adapter.Storage) (*MemoryTool, error)

// Name returns the name of the tool: "search_memory"
func (m *MemoryTool) Name() string

// Description returns a description for the LLM.
func (m *MemoryTool) Description() string

// Declaration returns the function declaration for the LLM.
func (m *MemoryTool) Declaration() *genai.FunctionDeclaration

// Call executes the memory search.
func (m *MemoryTool) Call(ctx context.Context, argsJSON string) (string, error)

// ProcessRequest implements toolinternal.RequestProcessor.
func (m *MemoryTool) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error
```

### Storage (Adapter)

The `Storage` interface defines the contract for observation storage backends.

```go
// Storage defines the interface for observation storage backends.
type Storage interface {
    // Store saves an observation to storage.
    Store(ctx context.Context, obs *Observation) error

    // GetByID retrieves an observation by its ID.
    GetByID(ctx context.Context, id string) (*Observation, error)

    // Search finds observations matching the given options.
    Search(ctx context.Context, opts *SearchOptions) ([]SearchResult, error)

    // Forget deletes an observation by ID.
    Forget(ctx context.Context, id string) error

    // Purge deletes observations matching the filter.
    Purge(ctx context.Context, filter map[string]string) error

    // IncrementTimesDerived increments the times_derived counter for an observation.
    IncrementTimesDerived(ctx context.Context, id string) error

    // Close releases any resources held by the storage.
    Close() error

    // QueryMostDerived returns observations sorted by times_derived DESC.
    QueryMostDerived(ctx context.Context, sessionID, userID, appName string, limit int) ([]Observation, error)

    // QueryRecent returns observations sorted by created_at DESC.
    QueryRecent(ctx context.Context, sessionID, userID, appName string, limit int) ([]Observation, error)
}

// SearchMode indicates which search strategy to use.
const (
    SearchModeVector SearchMode = iota  // Vector similarity search
    SearchModeFTS                        // Full-text search
    SearchModeHybrid                     // Reciprocal Rank Fusion of both
)

// Observation represents a single extracted fact.
type Observation struct {
    ID           string
    Content      string
    Level        ObservationLevel  // explicit, deductive, inductive, contradiction
    SessionID    string
    UserID       string
    AppName      string
    Tags         []string
    TimesDerived int
    CreatedAt    time.Time
    Embedding    []float32
}

// Score returns a composite score based on level and times_derived.
func (o Observation) Score() float64
```

### Storage Implementations

```go
// InMemory creates an in-memory SQLite storage (for testing).
func InMemory() (Storage, error)

// NewSQLiteStorage creates a file-based SQLite storage (for production).
func NewSQLiteStorage(path string) (Storage, error)
```

## Agent Integration Examples

### Basic Agent with Memory

A single agent that automatically extracts and recalls facts:

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"
    "strings"

    "google.golang.org/genai"

    "google.golang.org/adk/agent"
    adkmemory "google.golang.org/adk/memory"

    memory "github.com/ieshan/adk-go-memory"
    "github.com/ieshan/adk-go-memory/adapter"
    adkagent "google.golang.org/adk/agent"
    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/model"
    "google.golang.org/adk/model/gemini"
    "google.golang.org/adk/runner"
    "google.golang.org/adk/session"
)

func main() {
    ctx := context.Background()

    // Setup in-memory SQLite storage
    storage, err := adapter.InMemory()
    if err != nil {
        log.Fatalf("Failed to create storage: %v", err)
    }
    defer storage.Close()

    // Initialize LLM
    modelLLM, err := createLLM(ctx)
    if err != nil {
        log.Fatalf("Failed to create LLM: %v", err)
    }

    // Create deriver for automatic fact extraction
    deriver := memory.NewDeriver(memory.DeriverConfig{
        LLM:     modelLLM,
        Storage: storage,
    })

    // Create memory service
    svc := memory.NewService(memory.ServiceConfig{
        Storage: storage,
        Deriver: deriver,
    })
    defer svc.Close()

    // Create agent with memory-aware instruction and callback
    agentInst, err := llmagent.New(llmagent.Config{
        Name:        "memory_assistant",
        Model:       modelLLM,
        Description: "An assistant that remembers facts about users",
        Instruction: "You are a helpful assistant. Use the conversation context to answer questions.",
        BeforeModelCallbacks: []llmagent.BeforeModelCallback{
            injectMemoryContext(svc),
        },
    })
    if err != nil {
        log.Fatalf("Failed to create agent: %v", err)
    }

    // Create runner with memory service
    appName := "memory_app"
    sessionService := session.InMemoryService()
    r, err := runner.New(runner.Config{
        AppName:           appName,
        Agent:             agentInst,
        SessionService:    sessionService,
        MemoryService:     svc,
        AutoCreateSession: true,
    })
    if err != nil {
        log.Fatalf("Failed to create runner: %v", err)
    }

    // Run the agent
    userID := "user1"
    msg := genai.NewContentFromText("My name is Alice and I love Go programming", genai.RoleUser)
    
    for event, err := range r.Run(ctx, userID, "", msg, adkagent.RunConfig{}) {
        if err != nil {
            log.Printf("Error: %v", err)
            continue
        }
        if event.LLMResponse.Content != nil {
            for _, part := range event.LLMResponse.Content.Parts {
                fmt.Print(part.Text)
            }
        }
    }
}

// injectMemoryContext returns a BeforeModelCallback that searches memory
// and injects relevant context into the system prompt.
func injectMemoryContext(svc *memory.Service) llmagent.BeforeModelCallback {
    return func(ctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
        // Find the most recent user content
        var userQuery string
        for i := len(req.Contents) - 1; i >= 0; i-- {
            content := req.Contents[i]
            if content.Role == genai.RoleUser && len(content.Parts) > 0 {
                userQuery = content.Parts[0].Text
                break
            }
        }

        if userQuery == "" {
            return nil, nil
        }

        // Search memory for relevant context
        searchResp, err := svc.SearchMemory(ctx, &adkmemory.SearchRequest{
            Query:   userQuery,
            UserID:  ctx.UserID(),
            AppName: ctx.AppName(),
        })
        if err != nil || len(searchResp.Memories) == 0 {
            return nil, nil
        }

        // Build and inject memory context
        var memContext strings.Builder
        memContext.WriteString("\n\n**Relevant Information from Memory:**\n")
        for _, mem := range searchResp.Memories {
            if mem.Content != nil && len(mem.Content.Parts) > 0 {
                memContext.WriteString("- " + mem.Content.Parts[0].Text + "\n")
            }
        }

        // Append to system instruction
        for _, content := range req.Contents {
            if content.Role == "system" || content.Role == genai.RoleModel {
                if len(content.Parts) > 0 {
                    content.Parts[0].Text += memContext.String()
                    break
                }
            }
        }

        return nil, nil
    }
}

func createLLM(ctx context.Context) (model.LLM, error) {
    if os.Getenv("GOOGLE_API_KEY") != "" {
        return gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
            APIKey: os.Getenv("GOOGLE_API_KEY"),
        })
    }
    return nil, fmt.Errorf("GOOGLE_API_KEY not set")
}
```

### Agent with Memory Tool

An agent that can explicitly search its memory using a tool:

```go
package main

import (
    "context"
    "log"

    memory "github.com/ieshan/adk-go-memory"
    "github.com/ieshan/adk-go-memory/adapter"
    adkagent "google.golang.org/adk/agent"
    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/model"
    "google.golang.org/adk/runner"
    "google.golang.org/adk/session"
    "google.golang.org/adk/tool"
)

func main() {
    ctx := context.Background()

    // Setup storage and memory service
    storage, err := adapter.InMemory()
    if err != nil {
        log.Fatalf("Failed to create storage: %v", err)
    }
    defer storage.Close()

    modelLLM := getLLM() // Your LLM initialization

    deriver := memory.NewDeriver(memory.DeriverConfig{
        LLM:     modelLLM,
        Storage: storage,
    })

    svc := memory.NewService(memory.ServiceConfig{
        Storage: storage,
        Deriver: deriver,
    })
    defer svc.Close()

    // Create memory search tool
    memTool, err := memory.NewMemoryTool(storage)
    if err != nil {
        log.Fatalf("Failed to create memory tool: %v", err)
    }

    // Create agent with memory tool
    agent, err := llmagent.New(llmagent.Config{
        Name:        "tool_enabled_assistant",
        Model:       modelLLM,
        Description: "Assistant with memory search capability",
        Instruction: `You are a helpful assistant with access to a memory tool.

When users ask about their preferences, past activities, or personal information,
use the search_memory tool to find relevant information from previous conversations.

To use the tool, call search_memory with a query describing what you're looking for.
Example queries: "user preferences", "user name", "user hobbies", "past activities"`,
        Tools: []tool.Tool{memTool},
    })
    if err != nil {
        log.Fatalf("Failed to create agent: %v", err)
    }

    // Create runner
    appName := "memory_tool_app"
    sessionService := session.InMemoryService()
    r, err := runner.New(runner.Config{
        AppName:           appName,
        Agent:             agent,
        SessionService:    sessionService,
        MemoryService:     svc,
        AutoCreateSession: true,
    })
    if err != nil {
        log.Fatalf("Failed to create runner: %v", err)
    }

    // Use runner...
    _ = r
}

func getLLM() model.LLM {
    // Return your LLM implementation
    return nil
}
```

### Parent Agent with Sub-Agents Sharing Memory

A parent agent that delegates to specialized sub-agents, all sharing the same memory:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "google.golang.org/genai"
    adkmemory "google.golang.org/adk/memory"

    memory "github.com/ieshan/adk-go-memory"
    "github.com/ieshan/adk-go-memory/adapter"
    adkagent "google.golang.org/adk/agent"
    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/model"
    "google.golang.org/adk/runner"
    "google.golang.org/adk/session"
)

func main() {
    ctx := context.Background()

    // Setup shared storage and memory service
    storage, err := adapter.InMemory()
    if err != nil {
        log.Fatalf("Failed to create storage: %v", err)
    }
    defer storage.Close()

    modelLLM := getLLM()

    // Create deriver for automatic fact extraction
    deriver := memory.NewDeriver(memory.DeriverConfig{
        LLM:     modelLLM,
        Storage: storage,
    })

    // Create shared memory service
    svc := memory.NewService(memory.ServiceConfig{
        Storage: storage,
        Deriver: deriver,
    })
    defer svc.Close()

    // Create sub-agents
    factRecorder := createFactRecorder(modelLLM)
    greeter := createGreeter(modelLLM, svc)

    // Create parent agent with sub-agents
    parentAgent, err := llmagent.New(llmagent.Config{
        Name:        "assistant",
        Model:       modelLLM,
        Description: "Main assistant that can record facts or greet users. Delegate to fact_recorder for recording facts, greeter for greetings.",
        Instruction: "You are a helpful assistant with two specialized sub-agents:\n" +
            "1. fact_recorder - Use when users share facts about themselves\n" +
            "2. greeter - Use for greetings and social interactions\n" +
            "\nDelegate to the appropriate sub-agent based on the user's message.",
        SubAgents: []adkagent.Agent{factRecorder, greeter},
    })
    if err != nil {
        log.Fatalf("Failed to create parent agent: %v", err)
    }

    // Create runner with SHARED memory service
    // All sub-agents benefit from the same memory context
    appName := "subagent_app"
    sessionService := session.InMemoryService()
    r, err := runner.New(runner.Config{
        AppName:           appName,
        Agent:             parentAgent,
        SessionService:    sessionService,
        MemoryService:     svc, // Shared across all agents
        AutoCreateSession: true,
    })
    if err != nil {
        log.Fatalf("Failed to create runner: %v", err)
    }

    // Use runner...
    _ = r
}

// createFactRecorder creates a sub-agent that records user facts.
func createFactRecorder(modelLLM model.LLM) adkagent.Agent {
    agent, err := llmagent.New(llmagent.Config{
        Name:        "fact_recorder",
        Model:       modelLLM,
        Description: "Records and acknowledges user facts.",
        Instruction: "When users share facts about themselves, acknowledge and confirm you've noted them. " +
            "Be warm and encouraging about sharing.",
    })
    if err != nil {
        log.Fatalf("Failed to create fact_recorder: %v", err)
    }
    return agent
}

// createGreeter creates a sub-agent that greets users using memory context.
// This sub-agent explicitly searches memory to personalize greetings.
func createGreeter(modelLLM model.LLM, svc *memory.Service) adkagent.Agent {
    agent, err := llmagent.New(llmagent.Config{
        Name:        "greeter",
        Model:       modelLLM,
        Description: "Greets users by name using memory context.",
        Instruction: "Greet the user warmly. If you know their name from context, use it! " +
            "Be friendly and personable.",
        BeforeModelCallbacks: []llmagent.BeforeModelCallback{
            func(ctx adkagent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
                // Search memory for user name
                searchResp, err := svc.SearchMemory(ctx, &adkmemory.SearchRequest{
                    Query:   "user name",
                    UserID:  ctx.UserID(),
                    AppName: ctx.AppName(),
                })
                if err != nil || len(searchResp.Memories) == 0 {
                    return nil, nil
                }

                // Inject memory into system context
                memContext := "\n\n**Known Facts:**\n"
                for _, mem := range searchResp.Memories {
                    if mem.Content != nil && len(mem.Content.Parts) > 0 {
                        memContext += "- " + mem.Content.Parts[0].Text + "\n"
                    }
                }

                // Find system content and append
                for _, content := range req.Contents {
                    if content.Role == "system" || content.Role == genai.RoleModel {
                        if len(content.Parts) > 0 {
                            content.Parts[0].Text += memContext
                            break
                        }
                    }
                }
                return nil, nil
            },
        },
    })
    if err != nil {
        log.Fatalf("Failed to create greeter: %v", err)
    }
    return agent
}

func getLLM() model.LLM {
    // Return your LLM implementation
    return nil
}
```

### Advanced: Multi-Agent System with Provider and Peer Cards

A sophisticated setup using the Provider for context assembly and PeerCards for user modeling:

```go
package main

import (
    "context"
    "log"

    memory "github.com/ieshan/adk-go-memory"
    "github.com/ieshan/adk-go-memory/adapter"
    adkagent "google.golang.org/adk/agent"
    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/model"
    "google.golang.org/adk/runner"
    "google.golang.org/adk/session"
)

func main() {
    ctx := context.Background()

    // Setup storage
    storage, err := adapter.NewSQLiteStorage("/data/agent_memory.db")
    if err != nil {
        log.Fatalf("Failed to create storage: %v", err)
    }
    defer storage.Close()

    modelLLM := getLLM()

    // Create embedding function for semantic search
    embeddingFunc := func(ctx context.Context, text string) ([]float32, error) {
        // Use your embedding model here
        return nil, nil
    }

    // Create all memory components
    deriver := memory.NewDeriver(memory.DeriverConfig{
        LLM:     modelLLM,
        Storage: storage,
    })

    summarizer := memory.NewSummarizer(memory.SummarizerConfig{
        LLM:           modelLLM,
        Storage:       storage,
        ShortInterval: 20,
        LongInterval:  60,
    })

    dialectic := memory.NewDialectic(memory.DialecticConfig{
        LLM:        modelLLM,
        Storage:    storage,
        MaxResults: 10,
    })

    // Create provider with full pipeline
    provider := memory.NewProvider(memory.ProviderConfig{
        Storage:       storage,
        Deriver:       deriver,
        Summarizer:    summarizer,
        Dialectic:     dialectic,
        EmbeddingFunc: embeddingFunc,
    })
    defer provider.Close()

    // Create memory service with provider
    svc := memory.NewService(memory.ServiceConfig{
        Storage:  storage,
        Deriver:  deriver,
        Provider: provider,
    })
    defer svc.Close()

    // Load peer card for user on session start
    userID := "user123"
    if err := provider.LoadPeerCardFromMemory(ctx, userID); err != nil {
        log.Printf("Failed to load peer card: %v", err)
    }

    // Access peer card
    peerCard := provider.GetOrCreatePeerCard(userID)
    log.Printf("Loaded %d facts for user %s", len(peerCard.Facts()), userID)

    // Query peer card by type or tag
    explicitFacts := peerCard.ByType(adapter.LevelExplicit)
    preferences := peerCard.ByTag("preference")

    // Create agent with comprehensive memory context
    agent, err := llmagent.New(llmagent.Config{
        Name:        "advanced_assistant",
        Model:       modelLLM,
        Description: "Assistant with full memory pipeline",
        Instruction: "You are a helpful assistant with comprehensive memory.",
        BeforeModelCallbacks: []llmagent.BeforeModelCallback{
            func(ctx adkagent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
                // Get comprehensive memory context from provider
                memContext, err := provider.GetMemoryContext(ctx, "user preferences", ctx.SessionID(), ctx.UserID(), ctx.AppName())
                if err != nil {
                    return nil, nil
                }

                // Get peer card facts
                pc := provider.GetOrCreatePeerCard(ctx.UserID())
                peerContext := pc.Render()

                // Combine contexts
                fullContext := memContext + "\n" + peerContext

                // Inject into request
                for _, content := range req.Contents {
                    if content.Role == "system" {
                        if len(content.Parts) > 0 {
                            content.Parts[0].Text += "\n\n**Memory Context:**\n" + fullContext
                            break
                        }
                    }
                }
                return nil, nil
            },
        },
    })
    if err != nil {
        log.Fatalf("Failed to create agent: %v", err)
    }

    // Create runner
    sessionService := session.InMemoryService()
    r, err := runner.New(runner.Config{
        AppName:           "advanced_app",
        Agent:             agent,
        SessionService:    sessionService,
        MemoryService:     svc,
        AutoCreateSession: true,
    })
    if err != nil {
        log.Fatalf("Failed to create runner: %v", err)
    }

    // Use runner...
    _ = r
}

func getLLM() model.LLM {
    // Return your LLM implementation
    return nil
}
```

## Architecture

### Core Components

```
┌─────────────────┐     ┌─────────────┐     ┌─────────────────┐
│   ADK Agent     │────▶│   Service   │────▶│    Deriver      │
│                 │     │             │     │  (LLM-powered   │
└─────────────────┘     └──────┬──────┘     │  extraction)    │
                                 │            └─────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │      Provider           │
                    │  (Context assembly)     │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   RepresentationManager │
                    │  ├─ Semantic Search    │
                    │  ├─ Most Derived       │
                    │  └─ Recent             │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Storage (SQLite)      │
                    │  ├─ vec0 (vectors)      │
                    │  ├─ fts5 (text)         │
                    │  └─ observations        │
                    └─────────────────────────┘
```

### Observation Levels

Observations are classified by confidence:

| Level | Score | Description |
|-------|-------|-------------|
| `explicit` | 0.9 | Directly stated facts |
| `deductive` | 0.7 | Logical inferences |
| `inductive` | 0.5 | Patterns from multiple observations |
| `contradiction` | 0.3 | Conflicting statements |

### Search Modes

```go
// Vector search (semantic similarity)
results, err := storage.Search(ctx, &adapter.SearchOptions{
    Embedding:  queryEmbedding,
    Mode:       adapter.SearchModeVector,
    MaxResults: 10,
})

// FTS search (text matching)
results, err := storage.Search(ctx, &adapter.SearchOptions{
    Query:      "user likes golang",
    Mode:       adapter.SearchModeFTS,
    MaxResults: 10,
})

// Hybrid search (RRF fusion of both)
results, err := storage.Search(ctx, &adapter.SearchOptions{
    Query:      "user likes golang",
    Embedding:  queryEmbedding,
    Mode:       adapter.SearchModeHybrid,
    MaxResults: 10,
})
```

### Working Representations

Working representations combine three strategies for context assembly:

```go
rm := memory.NewRepresentationManager(memory.RepresentationConfig{
    Storage:           storage,
    SemanticBudget:    5,  // Top semantic search results
    MostDerivedBudget: 5,  // Most referenced facts
    RecentBudget:      5,  // Most recent observations
    EmbeddingFunc:     embedFunc, // For semantic search
})

rep, err := rm.GetWorkingRepresentation(ctx, query, sessionID, userID, appName)
if err != nil {
    log.Fatal(err)
}

// Format for context injection
context := rep.Format()
// Output:
// [semantic] User likes Go programming
// [semantic] User works at Google
// [derived] User prefers backend development
// [recent] User asked about databases
```

### Peer Cards

User profiles with scored facts:

```go
provider := memory.NewProvider(memory.ProviderConfig{
    Storage: storage,
})

// Load user's peer card
pc := provider.GetOrCreatePeerCard("user-123")

// Add facts
pc.AddFact(memory.PeerFact{
    Content: "User prefers Python over Java",
    Score:   0.85,
    Type:    adapter.LevelExplicit,
    Tags:    []string{"preference", "language"},
})

// Query by type or tag
preferences := pc.ByType(adapter.LevelExplicit)
langFacts := pc.ByTag("language")

// Render formatted profile
fmt.Println(pc.Render())
```

## Addressing ADK-Go Memory Limitations

### ADK-Go Built-in Memory

The standard ADK-Go memory system provides:
- Session storage via `session.Service`
- Basic memory interface via `memory.Service`
- Simple key-value storage

### Limitations Addressed by adk-go-memory

| ADK-Go Limitation | adk-go-memory Solution |
|-------------------|------------------------|
| No automatic fact extraction | LLM-powered `Deriver` extracts observations from conversations |
| No semantic search | `sqlite-vec` vector similarity search with 1536-dim embeddings |
| No full-text search | `FTS5` virtual table for text search |
| No deduplication | Hybrid search detects duplicates; increments `TimesDerived` |
| No user profiles | `PeerCard` system tracks up to 40 facts per user |
| Simple context assembly | Multi-strategy `RepresentationManager` (semantic + derived + recent) |
| No observation confidence | Four-level confidence system (explicit → contradiction) |
| In-memory only | Persistent SQLite storage with in-memory option |
| No memory tools | `MemoryTool` implements ADK tool interface |

## Configuration

### Storage Options

```go
// In-memory (testing)
storage, _ := adapter.InMemory()

// File-based (production)
storage, _ := adapter.NewSQLiteStorage("/data/memory.db")
```

### Deriver Configuration

```go
deriver := memory.NewDeriver(memory.DeriverConfig{
    LLM:     llm,      // Any model.LLM implementation
    Storage: storage,
})
```

### Provider with Full Pipeline

```go
provider := memory.NewProvider(memory.ProviderConfig{
    Storage:       storage,
    Deriver:       deriver,
    Summarizer:    summarizer,    // Optional: periodic summaries
    Dialectic:     dialectic,     // Optional: LLM-powered Q&A
    EmbeddingFunc: embedFunc,     // Required for semantic search
})
```

## Testing

Run tests with CGO enabled:

```bash
# Run all tests
CGO_ENABLED=1 go test -v -tags=sqlite_fts5 ./...

# Run with race detection
CGO_ENABLED=1 go test -v -race -tags=sqlite_fts5 ./...

# Coverage report
CGO_ENABLED=1 go test -tags=sqlite_fts5 -coverprofile=coverage.out ./...
go tool cover -func=coverage.out
```

## Makefile Targets

```bash
make test          # Run tests
make test-race     # Run with race detector
make build         # Build all packages
make vet           # Run go vet
make check         # Run vet + test
make coverage      # Generate coverage report
make clean         # Clean artifacts
```

## Dependencies

- `github.com/asg017/sqlite-vec-go-bindings` - Vector search for SQLite
- `github.com/mattn/go-sqlite3` - SQLite driver
- `google.golang.org/adk` - ADK-Go framework
- `google.golang.org/genai` - Google GenAI SDK

## License

This project is licensed under the [Mozilla Public License 2.0](https://www.mozilla.org/MPL/2.0/). See the [LICENSE](LICENSE) file for the full license text.

## Contributing

Contributions welcome! Please ensure:
1. Tests pass: `CGO_ENABLED=1 go test ./...`
2. Code is formatted: `go fmt ./...`
3. Vet passes: `go vet ./...`
4. No Honcho references remain (this is an independent ADK-Go package)
