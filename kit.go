// Package memory provides a memory layer for ADK-Go agents.
//
// It includes observation extraction, storage, retrieval, and context assembly
// for agent memory management. The package implements the ADK-Go memory.Service
// interface and provides tools for automatic memory preloading and explicit
// memory search.
//
// Key components:
//   - MemoryKit: One-stop setup for ADK-Go v1.2.0+ integration with compaction support
//   - Service: Implements google.golang.org/adk/memory.Service
//   - Provider: Advanced memory with representation management and peer cards
//   - Deriver: LLM-powered fact extraction from conversations
//   - MemoryTool: Explicit memory search tool (LLM-callable)
//   - PreloadMemoryTool: Automatic memory injection into system instructions
//   - CompactionPlugin: Optional session event compaction for context window management
package memory

import (
	"context"
	"fmt"

	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/tool"

	"github.com/ieshan/adk-go-memory/adapter"
	"github.com/ieshan/adk-go-memory/compaction"
)

// Compile-time interface compliance checks.
var _ memory.Service = (*Service)(nil)

// MemoryKit is the one-stop setup for ADK-Go v1.2.0+ integration.
//
// The kit provides:
//   - Service: Implements google.golang.org/adk/memory.Service
//   - Provider: Advanced memory with representation management
//   - LoadTool: Explicit memory search tool (LLM-callable)
//   - PreloadTool: Automatic memory injection into system instructions
//   - Tools: Memory tools (search, preload)
//   - Plugin: Optional compaction plugin for session event compaction
//
// Example usage:
//
//	kit, err := memory.New(memory.KitConfig{
//	    Storage: storage,
//	    LLM:     llm,  // Creates Deriver internally
//	    Compaction: &compaction.Config{
//	        Strategy:   &compaction.SummarizationStrategy{LLM: llm},
//	        MaxEvents:  100,
//	        MaxTokens:  4000,
//	        KeepRecent: 20,
//	    },
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer kit.Close()
type MemoryKit struct {
	Service     memory.Service // Implements google.golang.org/adk/memory.Service
	Provider    *Provider      // Advanced memory with context assembly
	LoadTool    tool.Tool      // Explicit memory search tool
	PreloadTool tool.Tool      // Automatic memory injection
	Tools       []tool.Tool    // All memory tools
	Plugin      *plugin.Plugin // Optional: session compaction
}

// KitConfig configures the memory kit.
type KitConfig struct {
	// Required
	Storage adapter.Storage

	// Optional: LLM for observation extraction and compaction
	// If provided, a Deriver will be created internally
	LLM model.LLM

	// Optional: Semantic search
	EmbeddingFunc func(ctx context.Context, text string) ([]float32, error)

	// Optional: Compaction configuration
	// If nil, compaction is disabled
	Compaction *compaction.Config

	// Optional: Advanced features
	EnableDialectic bool // LLM-powered Q&A over memory
	EnablePeerCards bool // User modeling
	DeltaMode       bool // Only process new events in AddSessionToMemory
}

// New creates a fully configured memory system.
//
// Usage:
//
//	kit, err := memory.New(memory.KitConfig{
//	    Storage: storage,
//	    LLM:     llm,
//	    Compaction: &compaction.Config{
//	        Strategy:   &compaction.SummarizationStrategy{LLM: llm},
//	        MaxEvents:  100,
//	        MaxTokens:  4000,
//	        KeepRecent: 20,
//	    },
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer kit.Close()
//
//	runner.New(runner.Config{
//	    PluginConfig: runner.PluginConfig{
//	        Plugins: []*plugin.Plugin{kit.Plugin},
//	    },
//	})
func New(cfg KitConfig) (*MemoryKit, error) {
	// Validate required fields
	if cfg.Storage == nil {
		return nil, fmt.Errorf("memory: Storage is required")
	}

	// Create provider with representation management
	provider := NewProvider(ProviderConfig{
		Storage:       cfg.Storage,
		EmbeddingFunc: cfg.EmbeddingFunc,
	})

	// Create deriver if LLM is provided
	var deriver *Deriver
	if cfg.LLM != nil {
		deriver = NewDeriver(DeriverConfig{
			LLM:     cfg.LLM,
			Storage: cfg.Storage,
		})
	}

	// Create service that implements memory.Service with optional delta mode
	svc := NewService(ServiceConfig{
		Provider:           provider,
		Deriver:            deriver,
		EnableDeltaMode:    cfg.DeltaMode,
		CompactionStateKey: "adk_memory_compaction_state",
	})

	// Create tools
	loadTool := NewMemoryTool(provider)
	preloadTool := NewPreloadMemoryTool(provider)

	tools := []tool.Tool{loadTool, preloadTool}

	// Create compaction plugin if configured
	var compactionPlugin *plugin.Plugin
	if cfg.Compaction != nil {
		p, err := compaction.NewPlugin(cfg.Compaction)
		if err != nil {
			return nil, fmt.Errorf("memory: failed to create compaction plugin: %w", err)
		}
		compactionPlugin = p
	}

	return &MemoryKit{
		Service:     svc,
		Provider:    provider,
		LoadTool:    loadTool,
		PreloadTool: preloadTool,
		Tools:       tools,
		Plugin:      compactionPlugin,
	}, nil
}

// Close releases all resources held by the kit.
func (k *MemoryKit) Close() error {
	if k.Provider != nil {
		return k.Provider.Close()
	}
	return nil
}
