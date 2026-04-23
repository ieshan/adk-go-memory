package memory

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/ieshan/adk-go-memory/adapter"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestProvider_GetMemoryContext(t *testing.T) {
	ctx := context.Background()
	storage, _ := adapter.InMemory()
	defer storage.Close()

	// Store some observations
	obs1 := &adapter.Observation{
		ID: "obs-1", Content: "user prefers dark mode", Level: adapter.LevelExplicit,
		SessionID: "s1", UserID: "u1", AppName: "a1",
		TimesDerived: 1, CreatedAt: time.Now(),
		Embedding: fakeEmbedding("dark mode preference"),
	}
	obs2 := &adapter.Observation{
		ID: "obs-2", Content: "user works with Python", Level: adapter.LevelExplicit,
		SessionID: "s1", UserID: "u1", AppName: "a1",
		TimesDerived: 1, CreatedAt: time.Now(),
		Embedding: fakeEmbedding("python programming"),
	}
	if err := storage.Store(ctx, obs1); err != nil {
		t.Fatalf("Store() error = %v", err)
	}
	if err := storage.Store(ctx, obs2); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	provider := NewProvider(ProviderConfig{
		Storage:       storage,
		EmbeddingFunc: func(ctx context.Context, text string) ([]float32, error) { return fakeEmbedding(text), nil },
	})

	result, err := provider.GetMemoryContext(ctx, "dark mode", "s1", "u1", "a1")
	if err != nil {
		t.Fatalf("GetMemoryContext() error = %v", err)
	}
	if result == "" {
		t.Error("Expected non-empty memory context")
	}
}

func TestProvider_GetMemoryContext_NilStorage(t *testing.T) {
	provider := NewProvider(ProviderConfig{Storage: nil})

	result, err := provider.GetMemoryContext(context.Background(), "test", "s1", "u1", "a1")
	if err != nil {
		t.Fatalf("GetMemoryContext() with nil storage should not error, got: %v", err)
	}
	if result != "" {
		t.Errorf("Expected empty result with nil storage, got: %q", result)
	}
}

func TestProvider_GetOrCreatePeerCard(t *testing.T) {
	provider := NewProvider(ProviderConfig{Storage: nil})

	// First call creates the card
	pc1 := provider.GetOrCreatePeerCard("user-1")
	if pc1 == nil {
		t.Fatal("Expected non-nil peer card")
	}
	if pc1.PeerID() != "user-1" {
		t.Errorf("PeerID = %q, want 'user-1'", pc1.PeerID())
	}

	// Second call returns the same card
	pc2 := provider.GetOrCreatePeerCard("user-1")
	if pc1 != pc2 {
		t.Error("Expected same peer card instance for same peer ID")
	}

	// Different peer ID creates a new card
	pc3 := provider.GetOrCreatePeerCard("user-2")
	if pc3.PeerID() != "user-2" {
		t.Errorf("PeerID = %q, want 'user-2'", pc3.PeerID())
	}
}

func TestProvider_LoadPeerCardFromMemory(t *testing.T) {
	ctx := context.Background()
	storage, _ := adapter.InMemory()
	defer storage.Close()

	// Store observations for a user with different derivation counts
	observations := []*adapter.Observation{
		{ID: "obs-1", Content: "user likes coffee", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 1, CreatedAt: time.Now()},
		{ID: "obs-2", Content: "user prefers tea", Level: adapter.LevelDeductive, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 5, CreatedAt: time.Now()},
		{ID: "obs-3", Content: "user works remotely", Level: adapter.LevelInductive, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 3, CreatedAt: time.Now()},
	}
	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	provider := NewProvider(ProviderConfig{Storage: storage})

	if err := provider.LoadPeerCardFromMemory(ctx, "u1"); err != nil {
		t.Fatalf("LoadPeerCardFromMemory() error = %v", err)
	}

	pc := provider.GetOrCreatePeerCard("u1")
	facts := pc.Facts()
	if len(facts) == 0 {
		t.Error("Expected peer card to have facts loaded from storage")
	}

	// Verify facts are loaded with proper scores derived from Observation.Score()
	// (which combines level + times_derived), not just from search result scores
	for _, f := range facts {
		if f.Score <= 0 {
			t.Errorf("Fact score = %v, want > 0", f.Score)
		}
	}
}

func TestProvider_LoadPeerCardFromMemory_PrioritizesMostDerived(t *testing.T) {
	ctx := context.Background()
	storage, _ := adapter.InMemory()
	defer storage.Close()

	// Store more observations than maxPeerCardFacts (40) to test prioritization.
	// Create 45 observations: 5 with high TimesDerived and 40 with low TimesDerived.
	// The peer card should prefer the high-derivation ones since QueryMostDerived
	// returns observations sorted by times_derived DESC.
	for i := 0; i < 40; i++ {
		obs := &adapter.Observation{
			ID:           fmt.Sprintf("obs-low-%d", i),
			Content:      fmt.Sprintf("low-importance fact %d", i),
			Level:        adapter.LevelInductive,
			SessionID:    "s1",
			UserID:       "u1",
			AppName:      "a1",
			TimesDerived: 1,
			CreatedAt:    time.Now(),
		}
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}
	for i := 0; i < 5; i++ {
		obs := &adapter.Observation{
			ID:           fmt.Sprintf("obs-high-%d", i),
			Content:      fmt.Sprintf("high-importance fact %d", i),
			Level:        adapter.LevelExplicit,
			SessionID:    "s1",
			UserID:       "u1",
			AppName:      "a1",
			TimesDerived: 10,
			CreatedAt:    time.Now().Add(-time.Hour), // older but more derived
		}
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	provider := NewProvider(ProviderConfig{Storage: storage})

	if err := provider.LoadPeerCardFromMemory(ctx, "u1"); err != nil {
		t.Fatalf("LoadPeerCardFromMemory() error = %v", err)
	}

	pc := provider.GetOrCreatePeerCard("u1")
	facts := pc.Facts()

	// All 5 high-importance facts should be present (they have higher scores)
	highCount := 0
	for _, f := range facts {
		if strings.Contains(f.Content, "high-importance") {
			highCount++
		}
	}
	if highCount != 5 {
		t.Errorf("Expected 5 high-importance facts in peer card, got %d", highCount)
	}
}

func TestProvider_LoadPeerCardFromMemory_NilStorage(t *testing.T) {
	provider := NewProvider(ProviderConfig{Storage: nil})

	err := provider.LoadPeerCardFromMemory(context.Background(), "u1")
	if err != nil {
		t.Fatalf("LoadPeerCardFromMemory() with nil storage should not error, got: %v", err)
	}
}

func TestProvider_OnSessionStart(t *testing.T) {
	ctx := context.Background()
	storage, _ := adapter.InMemory()
	defer storage.Close()

	// Store an observation for the user
	obs := &adapter.Observation{
		ID: "obs-1", Content: "user likes hiking", Level: adapter.LevelExplicit,
		SessionID: "s1", UserID: "u1", AppName: "a1",
		TimesDerived: 1, CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	provider := NewProvider(ProviderConfig{Storage: storage})

	if err := provider.OnSessionStart(ctx, "s1", "u1", "a1"); err != nil {
		t.Fatalf("OnSessionStart() error = %v", err)
	}

	// Peer card should be loaded for the user
	pc := provider.GetOrCreatePeerCard("u1")
	if len(pc.Facts()) == 0 {
		t.Error("Expected peer card to be loaded on session start")
	}
}

func TestProvider_OnSessionStart_EmptyUserID(t *testing.T) {
	provider := NewProvider(ProviderConfig{Storage: nil})

	err := provider.OnSessionStart(context.Background(), "s1", "", "a1")
	if err != nil {
		t.Fatalf("OnSessionStart() with empty userID should not error, got: %v", err)
	}
}

func TestProvider_Close(t *testing.T) {
	storage, _ := adapter.InMemory()
	provider := NewProvider(ProviderConfig{Storage: storage})

	if err := provider.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestProvider_Close_NilStorage(t *testing.T) {
	provider := NewProvider(ProviderConfig{Storage: nil})

	if err := provider.Close(); err != nil {
		t.Fatalf("Close() with nil storage should not error, got: %v", err)
	}
}

func TestProvider_GetMemoryContext_FallbackToSearch(t *testing.T) {
	ctx := context.Background()
	storage, _ := adapter.InMemory()
	defer storage.Close()

	// Store an observation WITHOUT embedding — representation manager
	// won't find it via semantic search, so it should fall back to simple search.
	obs := &adapter.Observation{
		ID: "obs-1", Content: "user likes Rust", Level: adapter.LevelExplicit,
		SessionID: "s1", UserID: "u1", AppName: "a1",
		TimesDerived: 1, CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	// Provider with embedding func but no embeddings stored —
	// representation manager returns empty, should fall back to search
	provider := NewProvider(ProviderConfig{
		Storage:       storage,
		EmbeddingFunc: func(ctx context.Context, text string) ([]float32, error) { return fakeEmbedding(text), nil },
	})

	result, err := provider.GetMemoryContext(ctx, "Rust", "s1", "u1", "a1")
	if err != nil {
		t.Fatalf("GetMemoryContext() error = %v", err)
	}
	if result == "" {
		t.Error("Expected non-empty memory context from fallback search")
	}
}

func TestProvider_Integration(t *testing.T) {
	ctx := context.Background()
	storage, _ := adapter.InMemory()
	defer storage.Close()

	llm := &fakeLLM{
		responses: []model.LLMResponse{{
			Content: &genai.Content{
				Parts: []*genai.Part{{
					Text: `{"observations":[{"content":"user prefers Go","level":"explicit","tags":["preference"]}]}`,
				}},
			},
		}},
	}

	deriver := NewDeriver(DeriverConfig{LLM: llm, Storage: storage})
	provider := NewProvider(ProviderConfig{
		Storage:    storage,
		Deriver:    deriver,
		Summarizer: NewSummarizer(SummarizerConfig{LLM: llm, Storage: storage}),
	})

	// Simulate session start
	if err := provider.OnSessionStart(ctx, "sess-1", "u1", "app1"); err != nil {
		t.Fatalf("OnSessionStart() error = %v", err)
	}

	// Get memory context (should work even with empty storage)
	result, err := provider.GetMemoryContext(ctx, "preferences", "sess-1", "u1", "app1")
	if err != nil {
		t.Fatalf("GetMemoryContext() error = %v", err)
	}
	// Result may be empty since no observations stored yet, but should not error
	_ = result

	// Close
	if err := provider.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}
