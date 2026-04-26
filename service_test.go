package memory

import (
	"context"
	"iter"
	"testing"
	"time"

	"github.com/ieshan/adk-go-memory/adapter"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

func TestService_AddSessionToMemory_ExtractsObservations(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()

	llm := &fakeLLM{
		responses: []model.LLMResponse{{
			Content: &genai.Content{
				Parts: []*genai.Part{{
					Text: `{"observations":[{"content":"user is 25 years old","level":"explicit"}]}`,
				}},
			},
		}},
	}

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	provider := NewProvider(ProviderConfig{Storage: storage})
	svc := NewService(ServiceConfig{Provider: provider, Deriver: deriver})

	sess := &mockSession{
		id:      "sess-1",
		userID:  "u1",
		appName: "app1",
		events: []*session.Event{
			{
				LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromText("I'm 25 years old", genai.RoleUser),
				},
				Author:    "user",
				Timestamp: time.Now(),
			},
		},
	}

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
		ID:        "obs-1",
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

	sess := &mockSession{id: "s1"}
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

	llm := &fakeLLM{}
	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	svc := NewService(ServiceConfig{Storage: storage, Deriver: deriver})

	sess := &mockSession{id: "s1", events: nil}
	if err := svc.AddSessionToMemory(ctx, sess); err != nil {
		t.Fatalf("AddSessionToMemory() with nil events error = %v", err)
	}
}

func TestService_AddSessionToMemory_EmptyEvents(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	llm := &fakeLLM{}
	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	svc := NewService(ServiceConfig{Storage: storage, Deriver: deriver})

	sess := &mockSession{id: "s1", events: []*session.Event{}}
	if err := svc.AddSessionToMemory(ctx, sess); err != nil {
		t.Fatalf("AddSessionToMemory() with empty events error = %v", err)
	}

	// No LLM call should happen
	if len(llm.calls) != 0 {
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
		{ID: "obs-u1-a1", Content: "user1 app1 fact", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "app1", CreatedAt: time.Now()},
		{ID: "obs-u2-a1", Content: "user2 app1 fact", Level: adapter.LevelExplicit, SessionID: "s2", UserID: "u2", AppName: "app1", CreatedAt: time.Now()},
		{ID: "obs-u1-a2", Content: "user1 app2 fact", Level: adapter.LevelExplicit, SessionID: "s3", UserID: "u1", AppName: "app2", CreatedAt: time.Now()},
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

	llm := &fakeLLM{
		responses: []model.LLMResponse{{
			Content: &genai.Content{
				Parts: []*genai.Part{{
					Text: `{"observations":[{"content":"user likes both Go and Python","level":"explicit"}]}`,
				}},
			},
		}},
	}

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	svc := NewService(ServiceConfig{Storage: storage, Deriver: deriver})

	sess := &mockSession{
		id:      "sess-multi",
		userID:  "u1",
		appName: "app1",
		events: []*session.Event{
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
		},
	}

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

	llm := &fakeLLM{
		responses: []model.LLMResponse{{
			Content: &genai.Content{
				Parts: []*genai.Part{{
					Text: `{"observations":[{"content":"user said hello","level":"explicit"}]}`,
				}},
			},
		}},
	}

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	svc := NewService(ServiceConfig{Storage: storage, Deriver: deriver})

	sess := &mockSession{
		id:      "sess-skip",
		userID:  "u1",
		appName: "app1",
		events: []*session.Event{
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
		},
	}

	if err := svc.AddSessionToMemory(ctx, sess); err != nil {
		t.Fatalf("AddSessionToMemory() error = %v", err)
	}
}

// mockSession implements session.Session for testing.
type mockSession struct {
	id      string
	userID  string
	appName string
	events  []*session.Event
}

func (m *mockSession) ID() string                { return m.id }
func (m *mockSession) AppName() string           { return m.appName }
func (m *mockSession) UserID() string            { return m.userID }
func (m *mockSession) State() session.State      { return nil }
func (m *mockSession) Events() session.Events    { return mockEvents(m.events) }
func (m *mockSession) LastUpdateTime() time.Time { return time.Now() }

// mockEvents implements session.Events for testing.
type mockEvents []*session.Event

func (e mockEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, ev := range e {
			if !yield(ev) {
				return
			}
		}
	}
}

func (e mockEvents) Len() int { return len(e) }

func (e mockEvents) At(i int) *session.Event {
	if i >= 0 && i < len(e) {
		return e[i]
	}
	return nil
}
