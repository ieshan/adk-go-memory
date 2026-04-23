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

func TestDeriver_ExtractsObservations(t *testing.T) {
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
					Text: `{"observations":[{"content":"user is 25 years old","level":"explicit"},{"content":"user likes Go","level":"explicit"}]}`,
				}},
			},
		}},
	}

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	msgs := []TimestampedMessage{
		{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "I'm 25 and I love Go"}}}, At: time.Now()},
	}

	err = deriver.Derive(ctx, msgs, "sess-1", "u1", "app1")
	if err != nil {
		t.Fatalf("Derive() error = %v", err)
	}

	// Query storage to verify observations were stored
	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "years",
		MaxResults: 10,
		Mode:       adapter.SearchModeFTS,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	if len(results) == 0 {
		t.Error("Expected at least one observation stored")
	}
}

func TestDeriver_HandlesEmptyResponse(t *testing.T) {
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
					Text: `{"observations":[]}`,
				}},
			},
		}},
	}

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	err = deriver.Derive(ctx, []TimestampedMessage{}, "sess-1", "u1", "app1")
	if err != nil {
		t.Fatalf("Derive() error = %v", err)
	}

	// Should have no errors with empty messages
	if len(llm.calls) != 0 {
		t.Error("Expected no LLM call for empty messages")
	}
}

func TestDeriver_Deduplication(t *testing.T) {
	ctx := context.Background()
	storage, err := adapter.InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	// Pre-populate with existing observation
	existing := &adapter.Observation{
		ID:           "obs-1",
		Content:      "user likes Go programming",
		Level:        adapter.LevelExplicit,
		SessionID:    "sess-1",
		UserID:       "u1",
		AppName:      "app1",
		TimesDerived: 1,
		CreatedAt:    time.Now(),
	}
	if err := storage.Store(ctx, existing); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	llm := &fakeLLM{
		responses: []model.LLMResponse{{
			Content: &genai.Content{
				Parts: []*genai.Part{{
					Text: `{"observations":[{"content":"user likes Go","level":"explicit"}]}`,
				}},
			},
		}},
	}

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	msgs := []TimestampedMessage{
		{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "I like Go"}}}, At: time.Now()},
	}

	err = deriver.Derive(ctx, msgs, "sess-1", "u1", "app1")
	if err != nil {
		t.Fatalf("Derive() error = %v", err)
	}

	// Should have incremented TimesDerived, not created duplicate
	obs, err := storage.GetByID(ctx, "obs-1")
	if err != nil {
		t.Fatalf("GetByID() error = %v", err)
	}

	if obs.TimesDerived != 2 {
		t.Errorf("TimesDerived = %d, want 2", obs.TimesDerived)
	}
}

func TestDeriver_GeneratesUniqueIDs(t *testing.T) {
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
					Text: `{"observations":[{"content":"fact A","level":"explicit"},{"content":"fact B","level":"explicit"}]}`,
				}},
			},
		}},
	}

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	msgs := []TimestampedMessage{
		{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "test"}}}, At: time.Now()},
	}

	err = deriver.Derive(ctx, msgs, "sess-1", "u1", "app1")
	if err != nil {
		t.Fatalf("Derive() error = %v", err)
	}

	// Search for all observations
	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "",
		MaxResults: 10,
		Mode:       adapter.SearchModeFTS,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("Expected 2 observations, got %d", len(results))
	}

	// Check IDs are unique
	if results[0].Observation.ID == results[1].Observation.ID {
		t.Error("Observation IDs should be unique")
	}

	// Check IDs have obs- prefix
	if !strings.HasPrefix(results[0].Observation.ID, "obs-") {
		t.Errorf("ID should have 'obs-' prefix: %s", results[0].Observation.ID)
	}
}

func TestDeriver_SetsSessionID(t *testing.T) {
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
					Text: `{"observations":[{"content":"test fact","level":"explicit"}]}`,
				}},
			},
		}},
	}

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	msgs := []TimestampedMessage{
		{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "test"}}}, At: time.Now()},
	}

	err = deriver.Derive(ctx, msgs, "custom-session-id", "u1", "app1")
	if err != nil {
		t.Fatalf("Derive() error = %v", err)
	}

	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "test",
		MaxResults: 10,
		Mode:       adapter.SearchModeFTS,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 observation, got %d", len(results))
	}

	if results[0].Observation.SessionID != "custom-session-id" {
		t.Errorf("SessionID = %s, want 'custom-session-id'", results[0].Observation.SessionID)
	}
}

func TestDeriver_SetsUserIDAndAppName(t *testing.T) {
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
					Text: `{"observations":[{"content":"test fact","level":"explicit"}]}`,
				}},
			},
		}},
	}

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	msgs := []TimestampedMessage{
		{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "test"}}}, At: time.Now()},
	}

	err = deriver.Derive(ctx, msgs, "sess-1", "user-42", "my-app")
	if err != nil {
		t.Fatalf("Derive() error = %v", err)
	}

	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "test",
		MaxResults: 10,
		Mode:       adapter.SearchModeFTS,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("Expected 1 observation, got %d", len(results))
	}

	if results[0].Observation.UserID != "user-42" {
		t.Errorf("UserID = %s, want 'user-42'", results[0].Observation.UserID)
	}
	if results[0].Observation.AppName != "my-app" {
		t.Errorf("AppName = %s, want 'my-app'", results[0].Observation.AppName)
	}
}

func TestDeriver_LLMError(t *testing.T) {
	ctx := context.Background()
	storage, err := adapter.InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	llm := &fakeLLM{} // No responses configured -> returns error

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	msgs := []TimestampedMessage{
		{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "test"}}}, At: time.Now()},
	}

	err = deriver.Derive(ctx, msgs, "sess-1", "u1", "app1")
	if err == nil {
		t.Error("Expected error when LLM has no responses")
	}
}

func TestDeriver_NilLLM(t *testing.T) {
	ctx := context.Background()
	storage, _ := adapter.InMemory()
	defer storage.Close()

	deriver := NewDeriver(DeriverConfig{LLM: nil, Storage: storage})
	msgs := []TimestampedMessage{
		{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "test"}}}, At: time.Now()},
	}

	err := deriver.Derive(ctx, msgs, "sess-1", "u1", "app1")
	if err == nil {
		t.Fatal("Expected error for Derive with nil LLM")
	}
	if !strings.Contains(err.Error(), "LLM is required") {
		t.Errorf("Error = %q, want error mentioning 'LLM is required'", err.Error())
	}
}

func TestDeriver_MalformedJSON(t *testing.T) {
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
					Text: `this is not valid JSON`,
				}},
			},
		}},
	}

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	msgs := []TimestampedMessage{
		{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "test"}}}, At: time.Now()},
	}

	err = deriver.Derive(ctx, msgs, "sess-1", "u1", "app1")
	if err == nil {
		t.Error("Expected error for malformed JSON response")
	}
}

func TestDeriver_EmptyLLMResponse(t *testing.T) {
	ctx := context.Background()
	storage, err := adapter.InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	llm := &fakeLLM{
		responses: []model.LLMResponse{{
			Content: &genai.Content{
				Parts: []*genai.Part{{Text: ""}},
			},
		}},
	}

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	msgs := []TimestampedMessage{
		{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "test"}}}, At: time.Now()},
	}

	err = deriver.Derive(ctx, msgs, "sess-1", "u1", "app1")
	if err != nil {
		t.Errorf("Expected nil error for empty text response, got: %v", err)
	}

	// No observations should be stored
	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "test",
		MaxResults: 10,
		Mode:       adapter.SearchModeFTS,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 0 {
		t.Errorf("Expected 0 observations for empty LLM text, got %d", len(results))
	}
}

func TestDeriver_InvalidLevel(t *testing.T) {
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
					Text: `{"observations":[{"content":"test fact","level":"invalid_level"}]}`,
				}},
			},
		}},
	}

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	msgs := []TimestampedMessage{
		{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "test"}}}, At: time.Now()},
	}

	err = deriver.Derive(ctx, msgs, "sess-1", "u1", "app1")
	if err != nil {
		t.Fatalf("Derive() error = %v", err)
	}

	// Observation should be stored with the level defaulted to inductive
	// (not stored as the invalid raw string)
	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "test",
		MaxResults: 10,
		Mode:       adapter.SearchModeFTS,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("Expected 1 observation, got %d", len(results))
	}
	if results[0].Observation.Level != adapter.LevelInductive {
		t.Errorf("Level = %s, want %s (invalid levels should default to inductive)", results[0].Observation.Level, adapter.LevelInductive)
	}
}

func TestDeriver_NoFalsePositiveDedup(t *testing.T) {
	ctx := context.Background()
	storage, err := adapter.InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	// Pre-populate with observation about Go
	existing := &adapter.Observation{
		ID:           "obs-1",
		Content:      "user likes Go programming",
		Level:        adapter.LevelExplicit,
		SessionID:    "sess-1",
		UserID:       "u1",
		AppName:      "app1",
		TimesDerived: 1,
		CreatedAt:    time.Now(),
	}
	if err := storage.Store(ctx, existing); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	// LLM extracts an unrelated observation about Python
	llm := &fakeLLM{
		responses: []model.LLMResponse{{
			Content: &genai.Content{
				Parts: []*genai.Part{{
					Text: `{"observations":[{"content":"user prefers Python for data science","level":"explicit"}]}`,
				}},
			},
		}},
	}

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	msgs := []TimestampedMessage{
		{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "I use Python for data science"}}}, At: time.Now()},
	}

	err = deriver.Derive(ctx, msgs, "sess-1", "u1", "app1")
	if err != nil {
		t.Fatalf("Derive() error = %v", err)
	}

	// Original observation should NOT have been incremented
	obs, err := storage.GetByID(ctx, "obs-1")
	if err != nil {
		t.Fatalf("GetByID() error = %v", err)
	}
	if obs.TimesDerived != 1 {
		t.Errorf("TimesDerived = %d, want 1 (should not be deduplicated)", obs.TimesDerived)
	}
}

func TestDeriver_DeduplicationWithEmbedding(t *testing.T) {
	ctx := context.Background()
	storage, err := adapter.InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	// Pre-populate with observation about Go — include an embedding.
	// Use the same content text as the LLM will produce so that fakeEmbedding
	// generates matching vectors (fakeEmbedding is a hash, not semantically meaningful).
	existing := &adapter.Observation{
		ID:           "obs-1",
		Content:      "user enjoys Go programming",
		Level:        adapter.LevelExplicit,
		SessionID:    "sess-1",
		UserID:       "u1",
		AppName:      "app1",
		TimesDerived: 1,
		CreatedAt:    time.Now(),
		Embedding:    fakeEmbedding("user enjoys Go programming"),
	}
	if err := storage.Store(ctx, existing); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	// LLM extracts an observation with identical content (near-duplicate)
	llm := &fakeLLM{
		responses: []model.LLMResponse{{
			Content: &genai.Content{
				Parts: []*genai.Part{{
					Text: `{"observations":[{"content":"user enjoys Go programming","level":"explicit"}]}`,
				}},
			},
		}},
	}

	// Deriver WITH embedding function — vector dedup should work
	deriver := NewDeriver(DeriverConfig{
		LLM:     llm,
		Storage: storage,
		EmbeddingFunc: func(ctx context.Context, text string) ([]float32, error) {
			return fakeEmbedding(text), nil
		},
	})
	msgs := []TimestampedMessage{
		{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "I enjoy Go"}}}, At: time.Now()},
	}

	err = deriver.Derive(ctx, msgs, "sess-1", "u1", "app1")
	if err != nil {
		t.Fatalf("Derive() error = %v", err)
	}

	// Original observation should have TimesDerived incremented (dedup worked)
	obs, err := storage.GetByID(ctx, "obs-1")
	if err != nil {
		t.Fatalf("GetByID() error = %v", err)
	}
	if obs.TimesDerived != 2 {
		t.Errorf("TimesDerived = %d, want 2 (should be deduplicated with embedding)", obs.TimesDerived)
	}
}

func TestDeriver_NilLLMResponse(t *testing.T) {
	ctx := context.Background()
	storage, err := adapter.InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	llm := &fakeLLM{
		responses: []model.LLMResponse{{}}, // nil Content

	}

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	msgs := []TimestampedMessage{
		{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "test"}}}, At: time.Now()},
	}

	err = deriver.Derive(ctx, msgs, "sess-1", "u1", "app1")
	if err != nil {
		t.Errorf("Expected nil error for nil LLM response content, got: %v", err)
	}
}

func TestValidLevel(t *testing.T) {
	tests := []struct {
		level string
		want  bool
	}{
		{"explicit", true},
		{"deductive", true},
		{"inductive", true},
		{"contradiction", true},
		{"invalid_level", false},
		{"", false},
		{"Explicit", false}, // case-sensitive
		{"EXPLICIT", false},
	}
	for _, tt := range tests {
		t.Run(tt.level, func(t *testing.T) {
			if got := validLevel(tt.level); got != tt.want {
				t.Errorf("validLevel(%q) = %v, want %v", tt.level, got, tt.want)
			}
		})
	}
}
