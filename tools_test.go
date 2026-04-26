package memory

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/ieshan/adk-go-memory/adapter"
)

func TestNewMemoryTool(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()

	// Pre-populate observations
	observations := []*adapter.Observation{
		{ID: "obs-1", Content: "user likes Go", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 1, CreatedAt: time.Now()},
		{ID: "obs-2", Content: "user is 25", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 1, CreatedAt: time.Now()},
	}
	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	tool, err := NewMemoryTool(storage)
	if err != nil {
		t.Fatalf("NewMemoryTool() error = %v", err)
	}

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

func TestMemoryTool_Call_WithResults(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()

	// Pre-populate storage
	obs := &adapter.Observation{
		ID:        "obs-1",
		Content:   "user enjoys hiking on weekends",
		Level:     adapter.LevelExplicit,
		UserID:    "user1",
		AppName:   "test-app",
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	tool, err := NewMemoryTool(storage)
	if err != nil {
		t.Fatalf("NewMemoryTool() error = %v", err)
	}

	argsJSON := `{"query":"hiking"}`
	result, err := tool.Call(ctx, argsJSON)
	if err != nil {
		t.Fatalf("Call() error = %v", err)
	}

	var output SearchMemoryResults
	if err := json.Unmarshal([]byte(result), &output); err != nil {
		t.Fatalf("Unmarshal result error = %v", err)
	}

	if len(output.Observations) == 0 {
		t.Error("Expected at least one search result")
	}
	if output.Observations[0].Content != "user enjoys hiking on weekends" {
		t.Errorf("Content = %q, want 'user enjoys hiking on weekends'", output.Observations[0].Content)
	}
}

func TestMemoryTool_Call_NoResults(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()

	tool, err := NewMemoryTool(storage)
	if err != nil {
		t.Fatalf("NewMemoryTool() error = %v", err)
	}

	argsJSON := `{"query":"nonexistent"}`
	result, err := tool.Call(ctx, argsJSON)
	if err != nil {
		t.Fatalf("Call() error = %v", err)
	}

	var output SearchMemoryResults
	if err := json.Unmarshal([]byte(result), &output); err != nil {
		t.Fatalf("Unmarshal result error = %v", err)
	}

	if len(output.Observations) != 0 {
		t.Errorf("Expected 0 results for empty storage, got %d", len(output.Observations))
	}
}

func TestMemoryTool_Call_InvalidJSON(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()

	tool, err := NewMemoryTool(storage)
	if err != nil {
		t.Fatalf("NewMemoryTool() error = %v", err)
	}

	_, err = tool.Call(ctx, "not valid json")
	if err == nil {
		t.Error("Expected error for invalid JSON args")
	}
}

func TestMemoryTool_Call_EmptyQuery(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()

	obs := &adapter.Observation{
		ID:        "obs-1",
		Content:   "some content",
		Level:     adapter.LevelExplicit,
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	tool, err := NewMemoryTool(storage)
	if err != nil {
		t.Fatalf("NewMemoryTool() error = %v", err)
	}

	argsJSON := `{"query":""}`
	result, err := tool.Call(ctx, argsJSON)
	if err != nil {
		t.Fatalf("Call() error = %v", err)
	}

	// Empty query should still work (fallback to recent)
	var output SearchMemoryResults
	if err := json.Unmarshal([]byte(result), &output); err != nil {
		t.Fatalf("Unmarshal result error = %v", err)
	}
}

func TestMemoryTool_Call_WithMaxResults(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()

	for i := 0; i < 5; i++ {
		obs := &adapter.Observation{
			ID:        "obs-" + string(rune('a'+i)),
			Content:   "test observation " + string(rune('a'+i)),
			Level:     adapter.LevelExplicit,
			CreatedAt: time.Now(),
		}
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	tool, err := NewMemoryTool(storage)
	if err != nil {
		t.Fatalf("NewMemoryTool() error = %v", err)
	}

	argsJSON := `{"query":"test","max_results":2}`
	result, err := tool.Call(ctx, argsJSON)
	if err != nil {
		t.Fatalf("Call() error = %v", err)
	}

	var output SearchMemoryResults
	if err := json.Unmarshal([]byte(result), &output); err != nil {
		t.Fatalf("Unmarshal result error = %v", err)
	}

	if len(output.Observations) > 2 {
		t.Errorf("Expected at most 2 results, got %d", len(output.Observations))
	}
}
