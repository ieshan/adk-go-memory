// Package memory provides a memory layer for ADK-Go agents.
//
// It includes observation extraction, storage, retrieval, and context assembly
// for agent memory management. The package implements the ADK-Go memory.Service
// interface and provides tools for automatic memory preloading and explicit
// memory search.
//
// Key components:
//   - MemoryKit: One-stop setup for ADK-Go v1.2.0 integration
//   - Service: Implements google.golang.org/adk/memory.Service
//   - Provider: Advanced memory with representation management and peer cards
//   - Deriver: LLM-powered fact extraction from conversations
//   - MemoryTool: Explicit memory search tool (LLM-callable)
//   - PreloadMemoryTool: Automatic memory injection into system instructions
package memory

import (
	"context"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/tool"

	"github.com/ieshan/adk-go-memory/adapter"
)

// Compile-time interface compliance checks.
var _ memory.Service = (*Service)(nil)

// MemoryKit bundles all components for ADK-Go v1.2.0 integration.
// It provides a one-stop setup for the enhanced memory system with
// automatic preloading and explicit memory search capabilities.
type MemoryKit struct {
	Service     memory.Service
	Provider    *Provider
	LoadTool    tool.Tool // FunctionTool + RequestProcessor
	PreloadTool tool.Tool // RequestProcessor only
}

// MemoryKitConfig configures the memory kit.
type MemoryKitConfig struct {
	Storage       adapter.Storage
	Deriver       *Deriver
	EmbeddingFunc func(ctx context.Context, text string) ([]float32, error)
}

// NewMemoryKit creates a fully wired memory system compatible with ADK-Go v1.2.0.
//
// Usage:
//
//	kit, err := memory.NewMemoryKit(memory.MemoryKitConfig{
//	    Storage: storage,
//	    Deriver: deriver,
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer kit.Close()
//
//	agent, err := llmagent.New(llmagent.Config{
//	    Tools: []tool.Tool{kit.PreloadTool, kit.LoadTool},
//	    // ... other config
//	})
//
// The kit provides:
//   - Service: Implements google.golang.org/adk/memory.Service
//   - Provider: Advanced memory with representation management
//   - LoadTool: Explicit memory search tool (LLM-callable)
//   - PreloadTool: Automatic memory injection into system instructions
func NewMemoryKit(cfg MemoryKitConfig) (*MemoryKit, error) {
	// Create provider with representation management
	provider := NewProvider(ProviderConfig{
		Storage:       cfg.Storage,
		EmbeddingFunc: cfg.EmbeddingFunc,
	})

	// Create service that implements memory.Service
	svc := NewService(ServiceConfig{
		Provider: provider,
		Deriver:  cfg.Deriver,
	})

	// Create tools
	loadTool := NewMemoryTool(provider)
	preloadTool := NewPreloadMemoryTool(provider)

	return &MemoryKit{
		Service:     svc,
		Provider:    provider,
		LoadTool:    loadTool,
		PreloadTool: preloadTool,
	}, nil
}

// Close releases all resources held by the kit.
func (k *MemoryKit) Close() error {
	return k.Provider.Close()
}
