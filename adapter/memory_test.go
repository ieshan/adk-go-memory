package adapter

import (
	"context"
	"testing"
	"time"
)

func TestMemoryStorage_StoreGetByID_RoundTrip(t *testing.T) {
	ctx := context.Background()
	storage := InMemory()
	defer storage.Close()

	obs := &Observation{
		ID:           "obs-1",
		Content:      "test observation",
		Level:        LevelExplicit,
		SessionID:    "s1",
		UserID:       "u1",
		AppName:      "a1",
		Tags:         []string{"tag1", "tag2"},
		TimesDerived: 5,
		CreatedAt:    time.Now(),
	}

	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	result, err := storage.GetByID(ctx, "obs-1")
	if err != nil {
		t.Fatalf("GetByID() error = %v", err)
	}

	if result.ID != obs.ID {
		t.Errorf("ID = %q, want %q", result.ID, obs.ID)
	}
	if result.Content != obs.Content {
		t.Errorf("Content = %q, want %q", result.Content, obs.Content)
	}
}

func TestMemoryStorage_Store_DuplicateID(t *testing.T) {
	ctx := context.Background()
	storage := InMemory()
	defer storage.Close()

	obs := &Observation{
		ID:        "obs-dup",
		Content:   "first",
		Level:     LevelExplicit,
		SessionID: "s1",
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	dup := &Observation{
		ID:        "obs-dup",
		Content:   "second",
		Level:     LevelExplicit,
		SessionID: "s1",
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, dup); err == nil {
		t.Error("Expected error for duplicate ID")
	}
}

func TestMemoryStorage_Search(t *testing.T) {
	ctx := context.Background()
	storage := InMemory()
	defer storage.Close()

	observations := []*Observation{
		{ID: "obs-1", Content: "user enjoys hiking", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
		{ID: "obs-2", Content: "user works as engineer", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
		{ID: "obs-3", Content: "user likes pizza", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
	}

	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	results, err := storage.Search(ctx, &SearchOptions{
		Query:      "hiking",
		MaxResults: 10,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}
	if len(results) > 0 && results[0].Observation.ID != "obs-1" {
		t.Errorf("Expected obs-1, got %s", results[0].Observation.ID)
	}
}

func TestMemoryStorage_Search_FilterBySessionID(t *testing.T) {
	ctx := context.Background()
	storage := InMemory()
	defer storage.Close()

	observations := []*Observation{
		{ID: "obs-s1", Content: "session one fact", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
		{ID: "obs-s2", Content: "session two fact", Level: LevelExplicit, SessionID: "s2", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
	}

	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	results, err := storage.Search(ctx, &SearchOptions{
		Query:      "fact",
		MaxResults: 10,
		SessionID:  "s1",
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	for _, r := range results {
		if r.Observation.SessionID != "s1" {
			t.Errorf("Got result from session %s, want only s1", r.Observation.SessionID)
		}
	}
}

func TestMemoryStorage_Forget(t *testing.T) {
	ctx := context.Background()
	storage := InMemory()
	defer storage.Close()

	obs := &Observation{
		ID:        "obs-delete",
		Content:   "to be deleted",
		Level:     LevelExplicit,
		SessionID: "s1",
		CreatedAt: time.Now(),
	}

	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	if err := storage.Forget(ctx, "obs-delete"); err != nil {
		t.Fatalf("Forget() error = %v", err)
	}

	_, err := storage.GetByID(ctx, "obs-delete")
	if err == nil {
		t.Error("Expected error for deleted observation")
	}
}

func TestMemoryStorage_QueryMostDerived(t *testing.T) {
	ctx := context.Background()
	storage := InMemory()
	defer storage.Close()

	observations := []*Observation{
		{ID: "obs-1", Content: "low derived", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 1, CreatedAt: time.Now()},
		{ID: "obs-2", Content: "high derived", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 10, CreatedAt: time.Now().Add(-1 * time.Hour)},
		{ID: "obs-3", Content: "medium derived", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 5, CreatedAt: time.Now().Add(-2 * time.Hour)},
	}

	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	results, err := storage.QueryMostDerived(ctx, "s1", "u1", "a1", 10)
	if err != nil {
		t.Fatalf("QueryMostDerived error = %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(results))
	}

	// Should be ordered by TimesDerived DESC
	if results[0].ID != "obs-2" {
		t.Errorf("Expected obs-2 first, got %s", results[0].ID)
	}
}

func TestMemoryStorage_QueryRecent(t *testing.T) {
	ctx := context.Background()
	storage := InMemory()
	defer storage.Close()

	now := time.Now()
	observations := []*Observation{
		{ID: "obs-old", Content: "old", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: now.Add(-2 * time.Hour)},
		{ID: "obs-new", Content: "new", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: now},
		{ID: "obs-middle", Content: "middle", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: now.Add(-1 * time.Hour)},
	}

	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	results, err := storage.QueryRecent(ctx, "s1", "u1", "a1", 10)
	if err != nil {
		t.Fatalf("QueryRecent error = %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("Expected 3 results, got %d", len(results))
	}

	// Should be ordered by CreatedAt DESC (newest first)
	if results[0].ID != "obs-new" {
		t.Errorf("Expected obs-new first, got %s", results[0].ID)
	}
}

func TestMemoryStorage_IncrementTimesDerived(t *testing.T) {
	ctx := context.Background()
	storage := InMemory()
	defer storage.Close()

	obs := &Observation{
		ID:           "obs-increment",
		Content:      "test",
		Level:        LevelExplicit,
		SessionID:    "s1",
		TimesDerived: 1,
		CreatedAt:    time.Now(),
	}

	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	if err := storage.IncrementTimesDerived(ctx, "obs-increment"); err != nil {
		t.Fatalf("IncrementTimesDerived error = %v", err)
	}

	result, err := storage.GetByID(ctx, "obs-increment")
	if err != nil {
		t.Fatalf("GetByID error = %v", err)
	}

	if result.TimesDerived != 2 {
		t.Errorf("TimesDerived = %d, want 2", result.TimesDerived)
	}
}

func TestMemoryStorage_Purge(t *testing.T) {
	ctx := context.Background()
	storage := InMemory()
	defer storage.Close()

	for i, sessionID := range []string{"s1", "s2"} {
		obs := &Observation{
			ID:        "obs-purge-" + sessionID,
			Content:   "content",
			Level:     LevelExplicit,
			SessionID: sessionID,
			UserID:    "u1",
			AppName:   "a1",
			CreatedAt: time.Now(),
		}
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store(%d) error = %v", i, err)
		}
	}

	if err := storage.Purge(ctx, map[string]string{"session_id": "s1"}); err != nil {
		t.Fatalf("Purge() error = %v", err)
	}

	_, err := storage.GetByID(ctx, "obs-purge-s1")
	if err == nil {
		t.Error("Expected s1 observation to be purged")
	}

	_, err = storage.GetByID(ctx, "obs-purge-s2")
	if err != nil {
		t.Fatalf("s2 observation should still exist: %v", err)
	}
}
