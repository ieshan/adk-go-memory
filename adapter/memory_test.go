package adapter

import (
	"context"
	"testing"
	"time"

	"github.com/ieshan/idx"
)

func TestMemoryStorage_StoreGetByID_RoundTrip(t *testing.T) {
	ctx := context.Background()
	storage := InMemory()
	defer storage.Close()

	id := idx.NewID()
	obs := &Observation{
		ID:           id,
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

	result, err := storage.GetByID(ctx, id)
	if err != nil {
		t.Fatalf("GetByID() error = %v", err)
	}

	if result.ID != obs.ID {
		t.Errorf("ID = %v, want %v", result.ID, obs.ID)
	}
	if result.Content != obs.Content {
		t.Errorf("Content = %q, want %q", result.Content, obs.Content)
	}
}

func TestMemoryStorage_Store_DuplicateID(t *testing.T) {
	ctx := context.Background()
	storage := InMemory()
	defer storage.Close()

	id := idx.NewID()
	obs := &Observation{
		ID:        id,
		Content:   "first",
		Level:     LevelExplicit,
		SessionID: "s1",
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	dup := &Observation{
		ID:        id,
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

	id1, id2, id3 := idx.NewID(), idx.NewID(), idx.NewID()
	observations := []*Observation{
		{ID: id1, Content: "user enjoys hiking", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
		{ID: id2, Content: "user works as engineer", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
		{ID: id3, Content: "user likes pizza", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
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
	if len(results) > 0 && results[0].Observation.ID != id1 {
		t.Errorf("Expected %v, got %v", id1, results[0].Observation.ID)
	}
}

func TestMemoryStorage_Search_FilterBySessionID(t *testing.T) {
	ctx := context.Background()
	storage := InMemory()
	defer storage.Close()

	id1, id2 := idx.NewID(), idx.NewID()
	observations := []*Observation{
		{ID: id1, Content: "session one fact", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
		{ID: id2, Content: "session two fact", Level: LevelExplicit, SessionID: "s2", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
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

	id := idx.NewID()
	obs := &Observation{
		ID:        id,
		Content:   "to be deleted",
		Level:     LevelExplicit,
		SessionID: "s1",
		CreatedAt: time.Now(),
	}

	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	if err := storage.Forget(ctx, id); err != nil {
		t.Fatalf("Forget() error = %v", err)
	}

	_, err := storage.GetByID(ctx, id)
	if err == nil {
		t.Error("Expected error for deleted observation")
	}
}

func TestMemoryStorage_QueryMostDerived(t *testing.T) {
	ctx := context.Background()
	storage := InMemory()
	defer storage.Close()

	id1, id2, id3 := idx.NewID(), idx.NewID(), idx.NewID()
	observations := []*Observation{
		{ID: id1, Content: "low derived", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 1, CreatedAt: time.Now()},
		{ID: id2, Content: "high derived", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 10, CreatedAt: time.Now().Add(-1 * time.Hour)},
		{ID: id3, Content: "medium derived", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 5, CreatedAt: time.Now().Add(-2 * time.Hour)},
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
	if results[0].ID != id2 {
		t.Errorf("Expected %v first, got %v", id2, results[0].ID)
	}
}

func TestMemoryStorage_QueryRecent(t *testing.T) {
	ctx := context.Background()
	storage := InMemory()
	defer storage.Close()

	now := time.Now()
	idOld, idNew, idMiddle := idx.NewID(), idx.NewID(), idx.NewID()
	observations := []*Observation{
		{ID: idOld, Content: "old", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: now.Add(-2 * time.Hour)},
		{ID: idNew, Content: "new", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: now},
		{ID: idMiddle, Content: "middle", Level: LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: now.Add(-1 * time.Hour)},
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
	if results[0].ID != idNew {
		t.Errorf("Expected %v first, got %v", idNew, results[0].ID)
	}
}

func TestMemoryStorage_IncrementTimesDerived(t *testing.T) {
	ctx := context.Background()
	storage := InMemory()
	defer storage.Close()

	id := idx.NewID()
	obs := &Observation{
		ID:           id,
		Content:      "test",
		Level:        LevelExplicit,
		SessionID:    "s1",
		TimesDerived: 1,
		CreatedAt:    time.Now(),
	}

	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	if err := storage.IncrementTimesDerived(ctx, id); err != nil {
		t.Fatalf("IncrementTimesDerived error = %v", err)
	}

	result, err := storage.GetByID(ctx, id)
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

	ids := []idx.ID{idx.NewID(), idx.NewID()}
	for i, sessionID := range []string{"s1", "s2"} {
		obs := &Observation{
			ID:        ids[i],
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

	_, err := storage.GetByID(ctx, ids[0])
	if err == nil {
		t.Error("Expected s1 observation to be purged")
	}

	_, err = storage.GetByID(ctx, ids[1])
	if err != nil {
		t.Fatalf("s2 observation should still exist: %v", err)
	}
}
