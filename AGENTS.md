# AGENTS.md

Guide for AI coding agents working in `github.com/ieshan/adk-go-memory`.

## Repository Snapshot

- Language: Go (`go 1.26` in root and submodules)
- Workspace: `go.work` includes:
  - root module (`.`)
  - `adapter/sqlite` submodule
  - `examples` submodule
- Main package purpose: ADK-Go memory service with observation extraction, storage/search, context assembly, summaries, and tool integration.

## Rule Files Found

- Found: `AGENTS.md` (this file), `MEMORY.md`
- Not found during scan: `.cursor/rules/*.md`, `.cursorrules`, `.github/copilot-instructions.md`, `claude.md`, lowercase `agents.md`

## Essential Commands

Use the Makefile at repo root.

```bash
# Core tests (root module)
make test

# SQLite adapter tests (submodule, CGO + sqlite_fts5 tag)
make test-sqlite

# Race tests across root + sqlite submodule
make test-race

# Build root + sqlite submodule
make build

# Vet root + sqlite submodule
make vet

# Full check
make check

# Coverage (root + sqlite adapter)
make coverage

# Clean coverage artifacts
make clean
```

Direct commands used by Makefile:

```bash
go test -v ./...
cd adapter/sqlite && CGO_ENABLED=1 go test -v -tags=sqlite_fts5 ./...
go test -v -race ./...
cd adapter/sqlite && CGO_ENABLED=1 go test -v -race -tags=sqlite_fts5 ./...
go build ./...
cd adapter/sqlite && CGO_ENABLED=1 go build -tags=sqlite_fts5 ./...
go vet ./...
cd adapter/sqlite && CGO_ENABLED=1 go vet -tags=sqlite_fts5 ./...
```

## Build/Test Requirements

- Root module is pure Go and uses `adapter.InMemory()` for storage tests.
- SQLite adapter in `adapter/sqlite` requires:
  - `CGO_ENABLED=1`
  - build tag `sqlite_fts5`
  - dependencies in submodule `adapter/sqlite/go.mod` (`mattn/go-sqlite3`, `sqlite-vec-go-bindings`)

## Project Structure

```text
.
├── service.go                  # memory.Service implementation
├── deriver.go                  # LLM-based fact extraction + dedup
├── provider.go                 # Orchestration + peer cards + representation manager
├── representation.go           # Semantic/derived/recent context assembly
├── peercard.go                 # Per-user scored facts (max 40)
├── summarizer.go               # Interval-based summaries
├── dialectic.go                # LLM answer synthesis from memory search
├── tools.go                    # search_memory ADK tool
├── adapter/
│   ├── adapter.go              # Core types/interfaces (Storage, Observation, SearchMode)
│   ├── memory.go               # In-memory map implementation
│   └── sqlite/                 # Separate module, SQLite + vec + FTS5
├── examples/                   # Separate module with runnable demos + tests
└── Makefile
```

## Code Patterns and Conventions Observed

### Interfaces and compile-time checks

- Interface conformance checks are used, e.g.:
  - `var _ memory.Service = (*Service)(nil)`
  - `var _ adapter.Storage = (*SQLiteStorage)(nil)`

### Error style

- Errors are wrapped with operation context (`fmt.Errorf("component: op: %w", err)`).
- Storage layers return explicit not-found / validation errors for many operations.

### Search defaults and modes

- `SearchModeHybrid` is zero value (`iota` first), so unset mode defaults to hybrid.
- Sources used in `SearchResult.Source` include:
  - `rrf`, `vector`, `fts`, and `vector_fallback`

### In-memory storage behavior (`adapter/memory.go`)

- Uses `sync.RWMutex` + map.
- Search is case-insensitive substring matching on content.
- Vector-only mode with no query returns empty results.
- `Store` and `GetByID` clone observations (including tags/embeddings) to avoid external mutation.

### SQLite storage behavior (`adapter/sqlite/sqlite.go`)

- Schema includes `observations`, `observations_fts` (FTS5), `vec_observations` (vec0 float[1536]).
- `Store`, `Forget`, and `Purge` use transactions to keep main/FTS/vector tables consistent.
- Hybrid search uses Reciprocal Rank Fusion (constant `k=60`).
- FTS query is sanitized (`sanitizeFTS5Query`) and can fallback to recent results on syntax errors.

### Deriver behavior (`deriver.go`)

- Requires `LLM`; otherwise returns error.
- LLM response must be JSON with `observations` array.
- Unknown observation level defaults to `inductive`.
- Dedup search uses hybrid mode and thresholding by source (`rrf`, `fts`, `vector`).
- Optional `EmbeddingFunc` enables vector-based dedup; nil means text-only dedup path.

### Provider / representation behavior

- `Provider` initializes a `RepresentationManager` with default budgets of 5/5/5 when storage exists.
- Working representation combines semantic + most-derived + recent while deduplicating by observation ID.
- `WorkingRepresentation.Format()` prefixes entries as `[semantic]`, `[derived]`, `[recent]`.

### Peer card behavior (`peercard.go`)

- Hard capacity: `maxPeerCardFacts = 40`.
- At capacity, only replaces the lowest-score fact if incoming fact score is higher.
- `Render()` sorts by score descending and groups by observation type.

## Testing Approach Observed

- Tests are colocated `*_test.go` across root, adapter, submodule, and examples.
- Fake LLM implementations are used for deterministic testing:
  - root: `fakeLLM` in `deriver.go`
  - examples: `examples/internal/testutil/fakellm.go`
- SQLite tests use `InMemory()` storage from sqlite submodule and validate:
  - vector/fts/hybrid behavior
  - table sync on delete/purge
  - filter behavior and defaults
- Example tests validate tool calls and ADK runner flows without external API calls.

## Gotchas and Non-Obvious Behaviors

- CGO/tag requirements apply only to `adapter/sqlite` paths.
- `SearchMode` default is hybrid because `SearchModeHybrid` is zero value.
- SQLite vector search without embedding intentionally falls back to recent observations (`vector_fallback`).
- `Purge` behavior differs by backend:
  - SQLite: rejects unknown keys and requires at least one recognized key (`session_id`, `user_id`, `app_name`).
  - In-memory: rejects unknown keys, but empty filter map will match all and delete all observations.
- Both `Service.Close()` and `Provider.Close()` call `storage.Close()`. If both share the same storage instance, avoid double-closing in new code paths.
- Example `main` programs use real LLM setup placeholders or API-key checks; tests rely on fake LLMs instead.

## Practical Agent Workflow

1. Prefer `make test` for root-only changes.
2. If touching `adapter/sqlite`, run `make test-sqlite` (and usually `make check`).
3. If touching search or retrieval semantics, run relevant adapter tests plus service/provider tests.
4. Keep interface assertions, error wrapping style, and cloning/dedup patterns consistent.
5. For new storage behavior, update both root adapter tests and sqlite submodule tests when behavior should match.
