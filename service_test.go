package memory

import (
	"context"
	"testing"
	"time"

	"github.com/ieshan/adk-go-memory/adapter"
	"github.com/ieshan/adk-go-pkg/testutil"
	"github.com/ieshan/idx"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

func TestService_AddSessionToMemory_ExtractsObservations(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()

	llm := testutil.NewFakeLLM(model.LLMResponse{
		Content: &genai.Content{
			Parts: []*genai.Part{{
				Text: `{"observations":[{"content":"user is 25 years old","level":"explicit"}]}`,
			}},
		},
	})

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	provider := NewProvider(ProviderConfig{Storage: storage})
	svc := NewService(ServiceConfig{Provider: provider, Deriver: deriver})

	events := []*session.Event{
		{
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText("I'm 25 years old", genai.RoleUser),
			},
			Author:    "user",
			Timestamp: time.Now(),
		},
	}
	sess := testutil.NewFakeSession().
		WithID("sess-1").
		WithUserID("u1").
		WithAppName("app1").
		WithEvents(events...)

	if err := svc.AddSessionToMemory(ctx, sess); err != nil {
		t.Fatalf("AddSessionToMemory() error = %v", err)
	}

	// Verify observation was stored
	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "25",
		MaxResults: 10,
		Mode:       adapter.SearchModeFTS,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) == 0 {
		t.Error("Expected at least one observation stored after AddSessionToMemory")
	}
}

func TestService_SearchMemory_ReturnsResults(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()

	provider := NewProvider(ProviderConfig{Storage: storage})
	svc := NewService(ServiceConfig{Provider: provider})

	// Pre-populate storage
	obs := &adapter.Observation{
		ID:        idx.NewID(),
		Content:   "user likes Go",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		UserID:    "u1",
		AppName:   "app1",
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	resp, err := svc.SearchMemory(ctx, &memory.SearchRequest{
		Query:   "Go",
		UserID:  "u1",
		AppName: "app1",
	})
	if err != nil {
		t.Fatalf("SearchMemory() error = %v", err)
	}

	if len(resp.Memories) == 0 {
		t.Error("Expected at least one memory result")
	}
	if resp.Memories[0].Content.Parts[0].Text != "user likes Go" {
		t.Errorf("Memory content = %q, want 'user likes Go'", resp.Memories[0].Content.Parts[0].Text)
	}
}

func TestService_SearchMemory_NilStorage(t *testing.T) {
	ctx := context.Background()
	svc := NewService(ServiceConfig{})

	resp, err := svc.SearchMemory(ctx, &memory.SearchRequest{Query: "test"})
	if err != nil {
		t.Fatalf("SearchMemory() error = %v", err)
	}
	if len(resp.Memories) != 0 {
		t.Error("Expected no memories with nil storage")
	}
}

func TestService_AddSessionToMemory_NilDeriver(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	svc := NewService(ServiceConfig{Storage: storage}) // No deriver

	sess := testutil.NewFakeSession().WithID("s1")
	if err := svc.AddSessionToMemory(ctx, sess); err != nil {
		t.Fatalf("AddSessionToMemory() with nil deriver error = %v", err)
	}
}

func TestService_AddSessionToMemory_NilSession(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	svc := NewService(ServiceConfig{Storage: storage})
	if err := svc.AddSessionToMemory(ctx, nil); err != nil {
		t.Fatalf("AddSessionToMemory(nil) error = %v", err)
	}
}

func TestService_AddSessionToMemory_NilEvents(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	llm := testutil.NewFakeLLM()
	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	svc := NewService(ServiceConfig{Storage: storage, Deriver: deriver})

	sess := testutil.NewFakeSession().WithID("s1")
	if err := svc.AddSessionToMemory(ctx, sess); err != nil {
		t.Fatalf("AddSessionToMemory() with nil events error = %v", err)
	}
}

func TestService_AddSessionToMemory_EmptyEvents(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	llm := testutil.NewFakeLLM()
	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	svc := NewService(ServiceConfig{Storage: storage, Deriver: deriver})

	sess := testutil.NewFakeSession().WithID("s1")
	if err := svc.AddSessionToMemory(ctx, sess); err != nil {
		t.Fatalf("AddSessionToMemory() with empty events error = %v", err)
	}

	// No LLM call should happen
	if llm.CallCount() != 0 {
		t.Error("Expected no LLM call for empty events")
	}
}

func TestService_Close(t *testing.T) {
	storage := adapter.InMemory()
	svc := NewService(ServiceConfig{Storage: storage})
	if err := svc.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestService_SearchMemory_WithFilters(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()

	// Store observations for different users/apps
	observations := []*adapter.Observation{
		{ID: idx.NewID(), Content: "user1 app1 fact", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "app1", CreatedAt: time.Now()},
		{ID: idx.NewID(), Content: "user2 app1 fact", Level: adapter.LevelExplicit, SessionID: "s2", UserID: "u2", AppName: "app1", CreatedAt: time.Now()},
		{ID: idx.NewID(), Content: "user1 app2 fact", Level: adapter.LevelExplicit, SessionID: "s3", UserID: "u1", AppName: "app2", CreatedAt: time.Now()},
	}
	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	svc := NewService(ServiceConfig{Storage: storage})

	// Search scoped to user1 + app1
	resp, err := svc.SearchMemory(ctx, &memory.SearchRequest{
		Query:   "fact",
		UserID:  "u1",
		AppName: "app1",
	})
	if err != nil {
		t.Fatalf("SearchMemory() error = %v", err)
	}

	for _, m := range resp.Memories {
		if m.Content.Parts[0].Text != "user1 app1 fact" {
			t.Errorf("Got unexpected result: %s", m.Content.Parts[0].Text)
		}
	}
}

func TestService_AddSessionToMemory_MultiPartContent(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()

	llm := testutil.NewFakeLLM(model.LLMResponse{
		Content: &genai.Content{
			Parts: []*genai.Part{{
				Text: `{"observations":[{"content":"user likes both Go and Python","level":"explicit"}]}`,
			}},
		},
	})

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	svc := NewService(ServiceConfig{Storage: storage, Deriver: deriver})

	events := []*session.Event{
		{
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role:  "user",
					Parts: []*genai.Part{{Text: "I like Go"}, {Text: "and Python"}},
				},
			},
			Author:    "user",
			Timestamp: time.Now(),
		},
	}
	sess := testutil.NewFakeSession().
		WithID("sess-multi").
		WithUserID("u1").
		WithAppName("app1").
		WithEvents(events...)

	if err := svc.AddSessionToMemory(ctx, sess); err != nil {
		t.Fatalf("AddSessionToMemory() error = %v", err)
	}

	// Verify observation was stored
	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "both Go", // Substring that exists in "user likes both Go and Python"
		MaxResults: 10,
		Mode:       adapter.SearchModeFTS,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) == 0 {
		t.Fatal("Expected at least one observation stored from multi-part content")
	}
}

func TestService_AddSessionToMemory_SkipsEmptyParts(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()

	llm := testutil.NewFakeLLM(model.LLMResponse{
		Content: &genai.Content{
			Parts: []*genai.Part{{
				Text: `{"observations":[{"content":"user said hello","level":"explicit"}]}`,
			}},
		},
	})

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	svc := NewService(ServiceConfig{Storage: storage, Deriver: deriver})

	events := []*session.Event{
		{
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role:  "user",
					Parts: []*genai.Part{{Text: ""}, {Text: "hello"}, {Text: ""}},
				},
			},
			Author:    "user",
			Timestamp: time.Now(),
		},
	}
	sess := testutil.NewFakeSession().
		WithID("sess-skip").
		WithUserID("u1").
		WithAppName("app1").
		WithEvents(events...)

	if err := svc.AddSessionToMemory(ctx, sess); err != nil {
		t.Fatalf("AddSessionToMemory() error = %v", err)
	}
}

func TestService_AddSessionToMemory_DeltaMode_ProcessesOnlyNewEvents(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()

	llm := testutil.NewFakeLLM(model.LLMResponse{
		Content: &genai.Content{
			Parts: []*genai.Part{{
				Text: `{"observations":[{"content":"user fact","level":"explicit"}]}`,
			}},
		},
	})

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	provider := NewProvider(ProviderConfig{Storage: storage})

	// Create service with delta mode enabled
	svc := NewService(ServiceConfig{
		Provider:           provider,
		Deriver:            deriver,
		EnableDeltaMode:    true,
		CompactionStateKey: "test_compaction_state",
	})

	// Create session with state containing compaction marker
	state := testutil.NewFakeStateWithData(map[string]any{"test_compaction_state": map[string]interface{}{"last_compacted_index": 1}})

	events := []*session.Event{
		{
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("first message", genai.RoleUser)},
			Author:      "user",
			Timestamp:   time.Now(),
		},
		{
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("second message", genai.RoleUser)},
			Author:      "user",
			Timestamp:   time.Now(),
		},
		{
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("third message", genai.RoleUser)},
			Author:      "user",
			Timestamp:   time.Now(),
		},
	}

	sess := testutil.NewFakeSession().
		WithID("sess-delta").
		WithUserID("u1").
		WithAppName("app1").
		WithEvents(events...).
		WithState(state.Data)

	if err := svc.AddSessionToMemory(ctx, sess); err != nil {
		t.Fatalf("AddSessionToMemory() error = %v", err)
	}

	// Should only process events after index 1 (third message only)
	// In delta mode with last_compacted_index=1, only event at index 2 should be processed
	if llm.CallCount() != 1 {
		t.Errorf("Expected 1 LLM call for delta mode, got %d", llm.CallCount())
	}
}

func TestService_AddSessionToMemory_DeltaMode_NoCompactionState(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()

	llm := testutil.NewFakeLLM(model.LLMResponse{
		Content: &genai.Content{
			Parts: []*genai.Part{{
				Text: `{"observations":[{"content":"user fact","level":"explicit"}]}`,
			}},
		},
	})

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	provider := NewProvider(ProviderConfig{Storage: storage})

	// Create service with delta mode enabled but no compaction state
	svc := NewService(ServiceConfig{
		Provider:           provider,
		Deriver:            deriver,
		EnableDeltaMode:    true,
		CompactionStateKey: "test_compaction_state",
	})

	events := []*session.Event{
		{
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("first message", genai.RoleUser)},
			Author:      "user",
			Timestamp:   time.Now(),
		},
		{
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("second message", genai.RoleUser)},
			Author:      "user",
			Timestamp:   time.Now(),
		},
	}

	sess := testutil.NewFakeSession().
		WithID("sess-delta-no-state").
		WithUserID("u1").
		WithAppName("app1").
		WithEvents(events...)

	if err := svc.AddSessionToMemory(ctx, sess); err != nil {
		t.Fatalf("AddSessionToMemory() error = %v", err)
	}

	// Should process all events when no compaction state exists
	if llm.CallCount() != 1 {
		t.Errorf("Expected 1 LLM call for all events, got %d", llm.CallCount())
	}
}

func TestService_getLastCompactedIndex(t *testing.T) {
	tests := []struct {
		name       string
		stateValue interface{}
		wantIndex  int
	}{
		{
			name:       "map with last_compacted_index",
			stateValue: map[string]interface{}{"last_compacted_index": 5},
			wantIndex:  5,
		},
		{
			name:       "nil state",
			stateValue: nil,
			wantIndex:  -1,
		},
		{
			name:       "empty map",
			stateValue: map[string]interface{}{},
			wantIndex:  -1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			storage := adapter.InMemory()
			svc := NewService(ServiceConfig{
				Storage:            storage,
				CompactionStateKey: "test_state",
			})

			stateData := make(map[string]any)
			if tt.stateValue != nil {
				stateData["test_state"] = tt.stateValue
			}
			state := testutil.NewFakeStateWithData(stateData)

			sess := testutil.NewFakeSession().WithID("s1").WithState(state.Data)
			got := svc.getLastCompactedIndex(sess)
			if got != tt.wantIndex {
				t.Errorf("getLastCompactedIndex() = %d, want %d", got, tt.wantIndex)
			}
		})
	}
}

func TestService_getEventsSince(t *testing.T) {
	events := []*session.Event{
		{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("event 0", genai.RoleUser)}},
		{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("event 1", genai.RoleUser)}},
		{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("event 2", genai.RoleUser)}},
		{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("event 3", genai.RoleUser)}},
	}

	storage := adapter.InMemory()
	svc := NewService(ServiceConfig{Storage: storage})

	sess := testutil.NewFakeSession().WithID("s1").WithEvents(events...)

	tests := []struct {
		name       string
		startIndex int
		wantCount  int
	}{
		{
			name:       "start at -1 (all events)",
			startIndex: -1,
			wantCount:  4,
		},
		{
			name:       "start at 1",
			startIndex: 1,
			wantCount:  2, // events 2 and 3
		},
		{
			name:       "start at 3 (last event)",
			startIndex: 3,
			wantCount:  0, // no events after index 3
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := svc.getEventsSince(sess, tt.startIndex)
			if len(got) != tt.wantCount {
				t.Errorf("getEventsSince() returned %d events, want %d", len(got), tt.wantCount)
			}
		})
	}
}
