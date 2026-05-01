package memory

import (
	"context"
	"testing"
	"time"

	"github.com/ieshan/adk-go-memory/adapter"
	"github.com/ieshan/adk-go-pkg/testutil"
	"github.com/ieshan/idx"
)

func TestRepresentationManager_StrategyOrdering(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()

	now := time.Now()
	// Create observations with specific TimesDerived and timestamps
	idRecentLow := idx.NewID()
	idOldHigh := idx.NewID()
	idMiddleMedium := idx.NewID()
	observations := []*adapter.Observation{
		{
			ID:           idRecentLow,
			Content:      "recent but low derived",
			Level:        adapter.LevelExplicit,
			SessionID:    "s1",
			UserID:       "u1",
			AppName:      "a1",
			TimesDerived: 1,
			CreatedAt:    now,
		},
		{
			ID:           idOldHigh,
			Content:      "old but high derived",
			Level:        adapter.LevelExplicit,
			SessionID:    "s1",
			UserID:       "u1",
			AppName:      "a1",
			TimesDerived: 20,
			CreatedAt:    now.Add(-24 * time.Hour),
		},
		{
			ID:           idMiddleMedium,
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

	// Check that most-derived section contains the high-derived observation
	// Derived observations start after semantic observations
	derivedStart := rep.SemanticCount
	foundHighDerived := false
	for i := derivedStart; i < derivedStart+rep.DerivedCount && i < len(rep.Observations); i++ {
		if rep.Observations[i].ID == idOldHigh {
			foundHighDerived = true
			break
		}
	}
	if !foundHighDerived {
		t.Error("Expected high-derived observation in derived observations")
	}

	// Check that recent section contains the recent observation
	recentStart := rep.SemanticCount + rep.DerivedCount
	foundRecent := false
	for i := recentStart; i < recentStart+rep.RecentCount && i < len(rep.Observations); i++ {
		if rep.Observations[i].ID == idRecentLow {
			foundRecent = true
			break
		}
	}
	if !foundRecent {
		t.Error("Expected recent observation in recent observations")
	}
}

func TestRepresentationManager_NoDuplicates(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()

	now := time.Now()
	// Single observation that would appear in all three strategies
	obsID := idx.NewID()
	obs := &adapter.Observation{
		ID:           obsID,
		Content:      "the only observation",
		Level:        adapter.LevelExplicit,
		SessionID:    "s1",
		UserID:       "u1",
		AppName:      "a1",
		TimesDerived: 100,
		Embedding:    testutil.FakeEmbed("the only observation"),
		CreatedAt:    now,
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	embedFn := func(ctx context.Context, text string) ([]float32, error) {
		return testutil.FakeEmbed("test"), nil
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
	ids := make(map[idx.ID]int)
	for _, o := range rep.Observations {
		ids[o.ID]++
	}
	for id, count := range ids {
		if count > 1 {
			t.Errorf("Observation %s appeared %d times, should be deduplicated", id.String(), count)
		}
	}

	// Total observations should be 1
	if len(rep.Observations) != 1 {
		t.Errorf("Expected 1 unique observation, got %d", len(rep.Observations))
	}
}

func TestRepresentationManager_EmptyStorage(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()

	embedFn := func(ctx context.Context, text string) ([]float32, error) {
		return testutil.FakeEmbed("test"), nil
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
	storage := adapter.InMemory()

	obs := &adapter.Observation{
		ID:           idx.NewID(),
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
			{ID: idx.NewID(), Content: "semantic fact"},
			{ID: idx.NewID(), Content: "derived fact"},
			{ID: idx.NewID(), Content: "recent fact"},
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
