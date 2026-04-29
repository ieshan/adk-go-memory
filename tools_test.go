package memory

import (
	"fmt"
	"testing"
	"time"

	"github.com/ieshan/adk-go-memory/adapter"
	"github.com/ieshan/adk-go-memory/internal/testutil"
	"github.com/ieshan/idx"
)

func TestNewMemoryTool(t *testing.T) {
	storage := adapter.InMemory()
	provider := NewProvider(ProviderConfig{Storage: storage})

	tool := NewMemoryTool(provider)

	if tool == nil {
		t.Fatal("NewMemoryTool() returned nil")
	}

	if tool.Name() != "search_memory" {
		t.Errorf("Name() = %v, want search_memory", tool.Name())
	}

	if tool.Description() == "" {
		t.Error("Expected non-empty description")
	}

	if tool.IsLongRunning() {
		t.Error("Expected IsLongRunning() to be false")
	}

	decl := tool.Declaration()
	if decl.Name != "search_memory" {
		t.Errorf("Declaration().Name = %q, want 'search_memory'", decl.Name)
	}
	if decl.Description == "" {
		t.Error("Declaration().Description should not be empty")
	}
}

func TestMemoryTool_Run_WithResults(t *testing.T) {
	storage := adapter.InMemory()

	// Pre-populate storage
	obs := &adapter.Observation{
		ID:        idx.NewID(),
		Content:   "user enjoys hiking on weekends",
		Level:     adapter.LevelExplicit,
		UserID:    "user1",
		AppName:   "test-app",
		CreatedAt: time.Now(),
	}
	if err := storage.Store(nil, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	provider := NewProvider(ProviderConfig{Storage: storage})
	tool := NewMemoryTool(provider)

	tc := testutil.NewFakeToolContext("user1", "test-app")
	args := map[string]any{"query": "hiking"}
	result, err := tool.Run(tc, args)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	observations, ok := result["observations"].([]ToolObservation)
	if !ok {
		t.Fatalf("Expected []ToolObservation, got %T", result["observations"])
	}

	if len(observations) == 0 {
		t.Error("Expected at least one search result")
	}
	if observations[0].Content != "user enjoys hiking on weekends" {
		t.Errorf("Content = %q, want 'user enjoys hiking on weekends'", observations[0].Content)
	}
}

func TestMemoryTool_Run_NoResults(t *testing.T) {
	storage := adapter.InMemory()
	provider := NewProvider(ProviderConfig{Storage: storage})
	tool := NewMemoryTool(provider)

	tc := testutil.NewFakeToolContext("user1", "test-app")
	args := map[string]any{"query": "nonexistent"}
	result, err := tool.Run(tc, args)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	observations, ok := result["observations"].([]ToolObservation)
	if !ok {
		t.Fatalf("Expected []ToolObservation, got %T", result["observations"])
	}

	if len(observations) != 0 {
		t.Errorf("Expected 0 results for empty storage, got %d", len(observations))
	}
}

func TestMemoryTool_Run_MissingQuery(t *testing.T) {
	storage := adapter.InMemory()
	provider := NewProvider(ProviderConfig{Storage: storage})
	tool := NewMemoryTool(provider)

	tc := testutil.NewFakeToolContext("user1", "test-app")
	args := map[string]any{"max_results": 5.0}
	_, err := tool.Run(tc, args)
	if err == nil {
		t.Error("Expected error for missing query")
	}
}

func TestMemoryTool_Run_EmptyQuery(t *testing.T) {
	storage := adapter.InMemory()

	obs := &adapter.Observation{
		ID:        idx.NewID(),
		Content:   "some content",
		Level:     adapter.LevelExplicit,
		CreatedAt: time.Now(),
	}
	if err := storage.Store(nil, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	provider := NewProvider(ProviderConfig{Storage: storage})
	tool := NewMemoryTool(provider)

	tc := testutil.NewFakeToolContext("user1", "test-app")
	args := map[string]any{"query": ""}
	result, err := tool.Run(tc, args)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	// Empty query should still work (fallback to recent)
	_, ok := result["observations"]
	if !ok {
		t.Error("Expected observations in result")
	}
}

func TestMemoryTool_Run_WithMaxResults(t *testing.T) {
	storage := adapter.InMemory()

	for i := 0; i < 5; i++ {
		obs := &adapter.Observation{
			ID:        idx.NewID(),
			Content:   fmt.Sprintf("test observation %d", i),
			Level:     adapter.LevelExplicit,
			CreatedAt: time.Now(),
		}
		if err := storage.Store(nil, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	provider := NewProvider(ProviderConfig{Storage: storage})
	tool := NewMemoryTool(provider)

	tc := testutil.NewFakeToolContext("user1", "test-app")
	args := map[string]any{"query": "test", "max_results": 2.0}
	result, err := tool.Run(tc, args)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	observations, ok := result["observations"].([]ToolObservation)
	if !ok {
		t.Fatalf("Expected []ToolObservation, got %T", result["observations"])
	}

	if len(observations) > 2 {
		t.Errorf("Expected at most 2 results, got %d", len(observations))
	}
}
