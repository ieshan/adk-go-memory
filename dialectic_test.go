package memory

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/ieshan/adk-go-memory/adapter"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestDialectic_Query(t *testing.T) {
	ctx := context.Background()
	storage, err := adapter.InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	// Pre-populate with observations
	observations := []*adapter.Observation{
		{ID: "obs-1", Content: "user is 25 years old", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 1, CreatedAt: time.Now()},
		{ID: "obs-2", Content: "user likes Go programming", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 1, CreatedAt: time.Now()},
	}
	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	llm := &fakeLLM{
		responses: []model.LLMResponse{{
			Content: &genai.Content{
				Parts: []*genai.Part{{
					Text: "The user is 25 years old and enjoys Go programming.",
				}},
			},
		}},
	}

	dialectic := NewDialectic(DialecticConfig{LLM: llm, Storage: storage, MaxResults: 10})
	answer, err := dialectic.Query(ctx, "Tell me about the user", QueryOptions{SessionID: "s1"})
	if err != nil {
		t.Fatalf("Query() error = %v", err)
	}

	if answer == "" {
		t.Error("Expected non-empty answer")
	}

	// Verify LLM was called with context
	if len(llm.calls) != 1 {
		t.Errorf("Expected 1 LLM call, got %d", len(llm.calls))
	}

	// Check that memory context was included
	foundContext := false
	for _, content := range llm.calls[0].Contents {
		for _, part := range content.Parts {
			if part.Text != "" && strings.Contains(part.Text, "Memory context:") {
				foundContext = true
				break
			}
		}
	}
	if !foundContext {
		t.Error("Expected memory context in LLM request")
	}
}

func TestDialectic_Query_NoResults(t *testing.T) {
	ctx := context.Background()
	storage, err := adapter.InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	llm := &fakeLLM{
		responses: []model.LLMResponse{{
			Content: &genai.Content{
				Parts: []*genai.Part{{
					Text: "I don't have any information about that.",
				}},
			},
		}},
	}

	dialectic := NewDialectic(DialecticConfig{LLM: llm, Storage: storage, MaxResults: 10})
	answer, err := dialectic.Query(ctx, "What is the user's favorite color?", QueryOptions{})
	if err != nil {
		t.Fatalf("Query() error = %v", err)
	}

	if answer == "" {
		t.Error("Expected non-empty answer even with no results")
	}
}

func TestDialectic_Query_DefaultMaxResults(t *testing.T) {
	storage, _ := adapter.InMemory()
	defer storage.Close()

	llm := &fakeLLM{}
	dialectic := NewDialectic(DialecticConfig{LLM: llm, Storage: storage})

	// Check default max results is 10
	if dialectic.maxResults != 10 {
		t.Errorf("maxResults = %d, want 10", dialectic.maxResults)
	}
}

func TestDialectic_Query_LLMError(t *testing.T) {
	ctx := context.Background()
	storage, _ := adapter.InMemory()
	defer storage.Close()

	// Pre-populate so search returns results, but LLM has no responses
	obs := &adapter.Observation{
		ID:        "obs-1",
		Content:   "test fact",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	llm := &fakeLLM{} // No responses -> error
	dialectic := NewDialectic(DialecticConfig{LLM: llm, Storage: storage})

	_, err := dialectic.Query(ctx, "test", QueryOptions{SessionID: "s1"})
	if err == nil {
		t.Error("Expected error when LLM has no responses")
	}
}

func TestDialectic_Query_NilStorage(t *testing.T) {
	ctx := context.Background()

	llm := &fakeLLM{}
	dialectic := NewDialectic(DialecticConfig{LLM: llm, Storage: nil})

	answer, err := dialectic.Query(ctx, "test", QueryOptions{})
	if err != nil {
		t.Fatalf("Query() with nil storage error = %v", err)
	}
	if answer != "" {
		t.Errorf("Expected empty answer with nil storage, got %q", answer)
	}
}

func TestDialectic_Query_NilLLMResponse(t *testing.T) {
	ctx := context.Background()
	storage, _ := adapter.InMemory()
	defer storage.Close()

	obs := &adapter.Observation{
		ID:        "obs-1",
		Content:   "test fact",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	// LLM returns nil Content
	llm := &fakeLLM{
		responses: []model.LLMResponse{{}},
	}
	dialectic := NewDialectic(DialecticConfig{LLM: llm, Storage: storage})

	answer, err := dialectic.Query(ctx, "test", QueryOptions{SessionID: "s1"})
	if err != nil {
		t.Fatalf("Query() error = %v", err)
	}
	if answer != "" {
		t.Errorf("Expected empty answer for nil LLM content, got %q", answer)
	}
}

func TestDialectic_Query_NilLLM(t *testing.T) {
	ctx := context.Background()
	storage, _ := adapter.InMemory()
	defer storage.Close()

	dialectic := NewDialectic(DialecticConfig{LLM: nil, Storage: storage})

	_, err := dialectic.Query(ctx, "test", QueryOptions{SessionID: "s1"})
	if err == nil {
		t.Fatal("Expected error for Query with nil LLM")
	}
	if !strings.Contains(err.Error(), "LLM is required") {
		t.Errorf("Error = %q, want error mentioning 'LLM is required'", err.Error())
	}
}
