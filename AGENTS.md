# AGENTS.md

Development guide for AI coding agents working on `adk-go-memory`.

## Project Overview

`adk-go-memory` is a production-grade memory layer for Google ADK-Go agents. It provides:

- Automatic observation extraction from conversations using LLM
- Hybrid search (vector + FTS5) via SQLite with sqlite-vec
- Deduplication via Reciprocal Rank Fusion
- Working representations for context assembly
- Peer Cards for user modeling
- Full ADK-Go `memory.Service` interface implementation

## Build Requirements

**Critical: CGO must be enabled for all operations.**

The project depends on `sqlite-vec` which requires CGO:

```bash
# All build/test commands require CGO
export CGO_ENABLED=1

# Build requires sqlite_fts5 build tag
go build -tags=sqlite_fts5 ./...
```

### Prerequisites

- Go 1.26+
- CGO-enabled toolchain
- SQLite with FTS5 support (via mattn/go-sqlite3)

## Development Commands

Use the Makefile for common tasks:

```bash
# Run tests (default)
make test

# Run with race detection
make test-race

# Build all packages
make build

# Run go vet
make vet

# Full check (vet + test)
make check

# Coverage report
make coverage

# Clean artifacts
make clean
```

## Testing

All tests require `CGO_ENABLED=1` and the `sqlite_fts5` build tag:

```bash
# Run all tests
CGO_ENABLED=1 go test -v -tags=sqlite_fts5 ./...

# Run specific package tests
CGO_ENABLED=1 go test -v -tags=sqlite_fts5 ./adapter

# Run with race detector
CGO_ENABLED=1 go test -v -race -tags=sqlite_fts5 ./...
```

### Test Structure

- `*_test.go` files alongside source files
- `fakeLLM` helper in `deriver.go` for mocking LLM calls
- In-memory SQLite storage used for isolated tests

## Project Structure

```
.
├── adapter/          # Storage abstraction and SQLite implementation
│   ├── adapter.go    # Storage interface and types
│   ├── sqlite.go     # SQLite + sqlite-vec implementation
│   └── sqlite_test.go
├── service.go        # Main memory.Service implementation
├── deriver.go        # LLM-powered observation extraction
├── provider.go       # Context assembly provider
├── representation.go # Working representation management
├── peercard.go       # User profile/fact tracking
├── summarizer.go     # Periodic conversation summarization
├── dialectic.go      # LLM-powered Q&A on memory
├── tools.go          # ADK tool integration
└── doc.go            # Package documentation
```

## Key Conventions

### Code Style

- Standard Go formatting: `go fmt ./...`
- Pass `context.Context` as first parameter
- Return errors with wrapped context: `fmt.Errorf("op: %w", err)`
- Interface compliance checks: `var _ Interface = (*Type)(nil)`

### Observation Levels

Observations have confidence levels defined in `adapter/adapter.go`:

- `explicit` (0.9) - Directly stated facts
- `deductive` (0.7) - Logical inferences
- `inductive` (0.5) - Patterns from 3+ observations
- `contradiction` (0.3) - Conflicting statements

### Storage Interface

All storage implementations must satisfy `adapter.Storage`:

```go
type Storage interface {
    Store(ctx context.Context, obs *Observation) error
    Search(ctx context.Context, opts *SearchOptions) ([]SearchResult, error)
    QueryMostDerived(ctx context.Context, sessionID, userID, appName string, limit int) ([]Observation, error)
    QueryRecent(ctx context.Context, sessionID, userID, appName string, limit int) ([]Observation, error)
    // ... see adapter/adapter.go
}
```

### Search Modes

The adapter supports three search modes:

- `SearchModeVector` - Vector similarity (requires embedding)
- `SearchModeFTS` - Full-text search via FTS5
- `SearchModeHybrid` - RRF fusion of both (default)

## Dependencies

Key external dependencies (see `go.mod`):

- `github.com/asg017/sqlite-vec-go-bindings` - Vector search for SQLite
- `github.com/mattn/go-sqlite3` - SQLite driver (requires CGO)
- `google.golang.org/adk` - ADK-Go framework
- `google.golang.org/genai` - Google GenAI SDK

## Common Tasks

### Adding a New Storage Backend

1. Implement `adapter.Storage` interface
2. Add constructor (e.g., `NewPostgresStorage`)
3. Create `*_test.go` with full test coverage
4. Update README.md with usage example

### Adding LLM-Powered Features

Follow the `Deriver` pattern:

1. Define config struct with `LLM model.LLM` and `Storage`
2. Create system prompt constant
3. Parse structured JSON from LLM responses
4. Add `fakeLLM` tests for deterministic testing

### Working with Observations

```go
// Creating an observation
obs := &adapter.Observation{
    ID:           generateID("obs"),
    Content:      "fact content",
    Level:        adapter.LevelExplicit,
    SessionID:    sessionID,
    Tags:         []string{"tag1", "tag2"},
    TimesDerived: 1,
    CreatedAt:    time.Now(),
}

// Score combines level + times_derived
score := obs.Score() // 0.0 - 1.0
```

## CI/PR Requirements

Before committing:

1. Run `make check` (vet + test)
2. Ensure `CGO_ENABLED=1` is set
3. All tests must pass with `-tags=sqlite_fts5`
4. Code formatted with `go fmt ./...`
5. No references to "Honcho" (this is an independent ADK-Go package)

## Important Notes

- **Never disable CGO** - sqlite-vec requires it
- **Always use build tag** - `-tags=sqlite_fts5` required for FTS5 virtual tables
- **Close storage** - Call `storage.Close()` to release resources
- **Session IDs** - Used for deduplication scoping
- **No breaking changes** to `memory.Service` interface (ADK-Go compatibility)
