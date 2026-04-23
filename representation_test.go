package memory

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"math"
	"testing"
	"time"

	"github.com/ieshan/adk-go-memory/adapter"
)

func TestRepresentationManager_StrategyOrdering(t *testing.T) {
	ctx := context.Background()
	storage, err := adapter.InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	now := time.Now()
	// Create observations with specific TimesDerived and timestamps
	observations := []*adapter.Observation{
		{
			ID:           "obs-recent-low",
			Content:      "recent but low derived",
			Level:        adapter.LevelExplicit,
			SessionID:    "s1",
			UserID:       "u1",
			AppName:      "a1",
			TimesDerived: 1,
			CreatedAt:    now,
		},
		{
			ID:           "obs-old-high",
			Content:      "old but high derived",
			Level:        adapter.LevelExplicit,
			SessionID:    "s1",
			UserID:       "u1",
			AppName:      "a1",
			TimesDerived: 20,
			CreatedAt:    now.Add(-24 * time.Hour),
		},
		{
			ID:           "obs-middle-medium",
			Content:      "middle age and medium derived",
			Level:        adapter.LevelExplicit,
			SessionID:    "s1",
			UserID:       "u1",
			AppName:      "a1",
			TimesDerived: 10,
			CreatedAt:    now.Add(-12 * time.Hour),
		},
	}

	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	// Create embedding function
	embedFn := func(ctx context.Context, text string) ([]float32, error) {
		// Return a dummy embedding
		vec := make([]float32, 1536)
		for i := range vec {
			vec[i] = 0.1
		}
		return vec, nil
	}

	rm := NewRepresentationManager(RepresentationConfig{
		Storage:           storage,
		SemanticBudget:    2,
		MostDerivedBudget: 2,
		RecentBudget:      2,
		EmbeddingFunc:     embedFn,
	})

	rep, err := rm.GetWorkingRepresentation(ctx, "test query", "s1", "u1", "a1")
	if err != nil {
		t.Fatalf("GetWorkingRepresentation error = %v", err)
	}

	// Check that most-derived section contains obs-old-high
	foundHighDerived := false
	for i := 0; i < rep.DerivedCount && i < len(rep.Observations); i++ {
		if rep.Observations[i].ID == "obs-old-high" {
			foundHighDerived = true
			break
		}
	}
	if !foundHighDerived {
		t.Error("Expected obs-old-high (highest TimesDerived) in derived observations")
	}

	// Check that recent section contains obs-recent-low
	recentStart := rep.SemanticCount + rep.DerivedCount
	foundRecent := false
	for i := recentStart; i < recentStart+rep.RecentCount && i < len(rep.Observations); i++ {
		if rep.Observations[i].ID == "obs-recent-low" {
			foundRecent = true
			break
		}
	}
	if !foundRecent {
		t.Error("Expected obs-recent-low (most recent) in recent observations")
	}
}

func TestRepresentationManager_NoDuplicates(t *testing.T) {
	ctx := context.Background()
	storage, err := adapter.InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	now := time.Now()
	// Single observation that would appear in all three strategies
	obs := &adapter.Observation{
		ID:           "obs-unique",
		Content:      "the only observation",
		Level:        adapter.LevelExplicit,
		SessionID:    "s1",
		UserID:       "u1",
		AppName:      "a1",
		TimesDerived: 100,
		Embedding:    fakeEmbedding("the only observation"),
		CreatedAt:    now,
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	embedFn := func(ctx context.Context, text string) ([]float32, error) {
		return fakeEmbedding("test"), nil
	}

	rm := NewRepresentationManager(RepresentationConfig{
		Storage:           storage,
		SemanticBudget:    5,
		MostDerivedBudget: 5,
		RecentBudget:      5,
		EmbeddingFunc:     embedFn,
	})

	rep, err := rm.GetWorkingRepresentation(ctx, "test query", "s1", "u1", "a1")
	if err != nil {
		t.Fatalf("GetWorkingRepresentation error = %v", err)
	}

	// The same observation should appear only once
	ids := make(map[string]int)
	for _, o := range rep.Observations {
		ids[o.ID]++
	}
	for id, count := range ids {
		if count > 1 {
			t.Errorf("Observation %s appeared %d times, should be deduplicated", id, count)
		}
	}

	// Total observations should be 1
	if len(rep.Observations) != 1 {
		t.Errorf("Expected 1 unique observation, got %d", len(rep.Observations))
	}
}

func TestRepresentationManager_EmptyStorage(t *testing.T) {
	ctx := context.Background()
	storage, err := adapter.InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	embedFn := func(ctx context.Context, text string) ([]float32, error) {
		return fakeEmbedding("test"), nil
	}

	rm := NewRepresentationManager(RepresentationConfig{
		Storage:           storage,
		SemanticBudget:    5,
		MostDerivedBudget: 5,
		RecentBudget:      5,
		EmbeddingFunc:     embedFn,
	})

	rep, err := rm.GetWorkingRepresentation(ctx, "test query", "s1", "u1", "a1")
	if err != nil {
		t.Fatalf("GetWorkingRepresentation error = %v", err)
	}

	if len(rep.Observations) != 0 {
		t.Errorf("Expected 0 observations from empty storage, got %d", len(rep.Observations))
	}
	if rep.SemanticCount != 0 || rep.DerivedCount != 0 || rep.RecentCount != 0 {
		t.Errorf("Expected all counts to be 0, got semantic=%d derived=%d recent=%d",
			rep.SemanticCount, rep.DerivedCount, rep.RecentCount)
	}
}

func TestRepresentationManager_NoEmbeddingFunc(t *testing.T) {
	ctx := context.Background()
	storage, err := adapter.InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	obs := &adapter.Observation{
		ID:           "obs-1",
		Content:      "test observation",
		Level:        adapter.LevelExplicit,
		SessionID:    "s1",
		UserID:       "u1",
		AppName:      "a1",
		TimesDerived: 5,
		CreatedAt:    time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	rm := NewRepresentationManager(RepresentationConfig{
		Storage:           storage,
		SemanticBudget:    5,
		MostDerivedBudget: 5,
		RecentBudget:      5,
		// No EmbeddingFunc
	})

	rep, err := rm.GetWorkingRepresentation(ctx, "test query", "s1", "u1", "a1")
	if err != nil {
		t.Fatalf("GetWorkingRepresentation error = %v", err)
	}

	// Should still get derived and recent results, just no semantic
	if rep.SemanticCount != 0 {
		t.Errorf("Expected 0 semantic results without embedding func, got %d", rep.SemanticCount)
	}
	if rep.DerivedCount+rep.RecentCount == 0 {
		t.Error("Expected some derived/recent results even without embedding func")
	}
}

func TestWorkingRepresentation_Format(t *testing.T) {
	rep := &WorkingRepresentation{
		Observations: []adapter.Observation{
			{ID: "obs-1", Content: "semantic fact"},
			{ID: "obs-2", Content: "derived fact"},
			{ID: "obs-3", Content: "recent fact"},
		},
		SemanticCount: 1,
		DerivedCount:  1,
		RecentCount:   1,
	}

	formatted := rep.Format()
	if !contains(formatted, "[semantic] semantic fact") {
		t.Error("Expected [semantic] section in Format output")
	}
	if !contains(formatted, "[derived] derived fact") {
		t.Error("Expected [derived] section in Format output")
	}
	if !contains(formatted, "[recent] recent fact") {
		t.Error("Expected [recent] section in Format output")
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		containsStr(s, substr))
}

func containsStr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// fakeEmbedding generates a deterministic 1536-dim float32 vector from text.
// Uses SHA-256 hash of text to seed the vector values for reproducible tests.
func fakeEmbedding(text string) []float32 {
	hash := sha256.Sum256([]byte(text))
	vec := make([]float32, 1536)

	for i := 0; i < 1536; i++ {
		idx := (i * 4) % len(hash)
		val := binary.BigEndian.Uint32(hash[idx:])
		vec[i] = (float32(val)/float32(math.MaxUint32))*2 - 1
	}

	return vec
}
