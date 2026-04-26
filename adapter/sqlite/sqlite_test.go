package sqlite

import (
	"context"
	"testing"
	"time"

	"github.com/ieshan/adk-go-memory/adapter"
)

func TestSQLiteStorage_VectorSearch(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	// Create observations with embeddings
	// "golang" and "programming" should be closer to "Go" query than "pizza"
	observations := []*adapter.Observation{
		{
			ID:        "obs-go",
			Content:   "user likes Go programming",
			Level:     adapter.LevelExplicit,
			SessionID: "s1",
			UserID:    "u1",
			AppName:   "a1",
			Embedding: fakeEmbedding("golang programming language"),
			CreatedAt: time.Now(),
		},
		{
			ID:        "obs-pizza",
			Content:   "user enjoys pizza",
			Level:     adapter.LevelExplicit,
			SessionID: "s1",
			UserID:    "u1",
			AppName:   "a1",
			Embedding: fakeEmbedding("italian food pizza cheese"),
			CreatedAt: time.Now(),
		},
		{
			ID:        "obs-py",
			Content:   "user knows Python",
			Level:     adapter.LevelExplicit,
			SessionID: "s1",
			UserID:    "u1",
			AppName:   "a1",
			Embedding: fakeEmbedding("python programming code"),
			CreatedAt: time.Now(),
		},
	}

	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	// Search with embedding similar to Go/programming
	queryEmbedding := fakeEmbedding("golang development")
	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "", // Text query not used in vector mode
		Embedding:  queryEmbedding,
		MaxResults: 3,
		Mode:       adapter.SearchModeVector,
		SessionID:  "s1",
		UserID:     "u1",
		AppName:    "a1",
	})
	if err != nil {
		t.Fatalf("Search(vector) error = %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Expected vector search results")
	}

	// Verify source is marked correctly
	if results[0].Source != "vector" {
		t.Errorf("Expected source='vector', got %s", results[0].Source)
	}

	// Verify score is positive and reasonable
	if results[0].Score <= 0 || results[0].Score > 1 {
		t.Errorf("Expected score in (0,1], got %f", results[0].Score)
	}

	// Pizza should not be the top result (it's semantically different)
	// Note: hash-based fake embeddings aren't truly semantic, so we only
	// check that results are returned and have valid properties.
	foundProgramming := false
	for _, r := range results {
		if r.Observation.ID == "obs-go" || r.Observation.ID == "obs-py" {
			foundProgramming = true
			break
		}
	}
	if !foundProgramming {
		t.Error("Expected at least one programming observation in results")
	}
}

func TestSQLiteStorage_HybridSearch(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	// Create observations - some match FTS better, some match vector better
	observations := []*adapter.Observation{
		{
			ID:           "obs-go-text",
			Content:      "Go is a programming language created at Google",
			Level:        adapter.LevelExplicit,
			SessionID:    "s1",
			UserID:       "u1",
			AppName:      "a1",
			Embedding:    fakeEmbedding("completely unrelated text about food"),
			TimesDerived: 1,
			CreatedAt:    time.Now().Add(-1 * time.Hour),
		},
		{
			ID:           "obs-go-vec",
			Content:      "The user mentioned they use Golang for backend work",
			Level:        adapter.LevelExplicit,
			SessionID:    "s1",
			UserID:       "u1",
			AppName:      "a1",
			Embedding:    fakeEmbedding("golang programming backend development"),
			TimesDerived: 5, // Highly derived
			CreatedAt:    time.Now(),
		},
	}

	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	queryEmbedding := fakeEmbedding("golang programming")
	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "Go programming",
		Embedding:  queryEmbedding,
		MaxResults: 5,
		Mode:       adapter.SearchModeHybrid,
		SessionID:  "s1",
		UserID:     "u1",
		AppName:    "a1",
	})
	if err != nil {
		t.Fatalf("Search(hybrid) error = %v", err)
	}

	if len(results) < 2 {
		t.Fatalf("Expected at least 2 hybrid results, got %d", len(results))
	}

	// Both observations should appear (RRF fusion should combine them)
	ids := make(map[string]bool)
	rrfCount := 0
	for _, r := range results {
		ids[r.Observation.ID] = true
		if r.Source == "rrf" {
			rrfCount++
		}
	}

	if !ids["obs-go-text"] {
		t.Error("Expected obs-go-text in hybrid results (FTS match)")
	}
	if !ids["obs-go-vec"] {
		t.Error("Expected obs-go-vec in hybrid results (vector match)")
	}
	if rrfCount == 0 {
		t.Error("Expected some results with source='rrf' from fusion")
	}
}

func TestSQLiteStorage_QueryMostDerived(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	observations := []*adapter.Observation{
		{ID: "obs-1", Content: "low derived", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 1, CreatedAt: time.Now()},
		{ID: "obs-2", Content: "high derived", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 10, CreatedAt: time.Now().Add(-1 * time.Hour)},
		{ID: "obs-3", Content: "medium derived", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 5, CreatedAt: time.Now().Add(-2 * time.Hour)},
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
	if results[0].ID != "obs-2" || results[0].TimesDerived != 10 {
		t.Errorf("Expected obs-2 (TimesDerived=10) first, got %s (TimesDerived=%d)", results[0].ID, results[0].TimesDerived)
	}
	if results[1].ID != "obs-3" || results[1].TimesDerived != 5 {
		t.Errorf("Expected obs-3 (TimesDerived=5) second, got %s (TimesDerived=%d)", results[1].ID, results[1].TimesDerived)
	}
	if results[2].ID != "obs-1" || results[2].TimesDerived != 1 {
		t.Errorf("Expected obs-1 (TimesDerived=1) third, got %s (TimesDerived=%d)", results[2].ID, results[2].TimesDerived)
	}
}

func TestSQLiteStorage_QueryRecent(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	now := time.Now()
	observations := []*adapter.Observation{
		{ID: "obs-old", Content: "old", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: now.Add(-2 * time.Hour)},
		{ID: "obs-new", Content: "new", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: now},
		{ID: "obs-middle", Content: "middle", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: now.Add(-1 * time.Hour)},
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
		t.Errorf("Expected obs-new first (newest), got %s", results[0].ID)
	}
	if results[1].ID != "obs-middle" {
		t.Errorf("Expected obs-middle second, got %s", results[1].ID)
	}
	if results[2].ID != "obs-old" {
		t.Errorf("Expected obs-old third (oldest), got %s", results[2].ID)
	}
}

func TestSQLiteStorage_Forget_SyncsVirtualTables(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	obs := &adapter.Observation{
		ID:        "obs-delete",
		Content:   "to be deleted",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		UserID:    "u1",
		AppName:   "a1",
		Embedding: fakeEmbedding("test content"),
		CreatedAt: time.Now(),
	}

	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	// Verify it exists in FTS
	ftsResults, _ := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "deleted",
		MaxResults: 1,
		Mode:       adapter.SearchModeFTS,
	})
	if len(ftsResults) == 0 {
		t.Fatal("Expected to find observation in FTS before deletion")
	}

	// Delete it
	if err := storage.Forget(ctx, "obs-delete"); err != nil {
		t.Fatalf("Forget() error = %v", err)
	}

	// Verify it's gone from main table
	_, err = storage.GetByID(ctx, "obs-delete")
	if err == nil {
		t.Error("Expected observation to be deleted from main table")
	}

	// Verify it's gone from FTS
	ftsResults, _ = storage.Search(ctx, &adapter.SearchOptions{
		Query:      "deleted",
		MaxResults: 1,
		Mode:       adapter.SearchModeFTS,
	})
	for _, r := range ftsResults {
		if r.Observation.ID == "obs-delete" {
			t.Error("Observation still found in FTS after deletion")
		}
	}
}

func TestSQLiteStorage_IncrementTimesDerived(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	obs := &adapter.Observation{
		ID:           "obs-increment",
		Content:      "test content",
		Level:        adapter.LevelExplicit,
		SessionID:    "s1",
		UserID:       "u1",
		AppName:      "a1",
		TimesDerived: 1,
		CreatedAt:    time.Now(),
	}

	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	// Increment times_derived
	if err := storage.IncrementTimesDerived(ctx, "obs-increment"); err != nil {
		t.Fatalf("IncrementTimesDerived error = %v", err)
	}

	// Verify
	result, err := storage.GetByID(ctx, "obs-increment")
	if err != nil {
		t.Fatalf("GetByID error = %v", err)
	}

	if result.TimesDerived != 2 {
		t.Errorf("TimesDerived = %d, want 2", result.TimesDerived)
	}
}

func TestSQLiteStorage_StoreGetByID_RoundTrip(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	now := time.Now().Truncate(time.Second) // Truncate for SQLite DATETIME precision
	obs := &adapter.Observation{
		ID:           "obs-full",
		Content:      "full round trip test",
		Level:        adapter.LevelDeductive,
		SessionID:    "s1",
		UserID:       "u1",
		AppName:      "a1",
		Tags:         []string{"tag1", "tag2"},
		TimesDerived: 5,
		CreatedAt:    now,
		Embedding:    fakeEmbedding("test embedding round trip"),
	}

	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	result, err := storage.GetByID(ctx, "obs-full")
	if err != nil {
		t.Fatalf("GetByID() error = %v", err)
	}

	if result.ID != obs.ID {
		t.Errorf("ID = %q, want %q", result.ID, obs.ID)
	}
	if result.Content != obs.Content {
		t.Errorf("Content = %q, want %q", result.Content, obs.Content)
	}
	if result.Level != obs.Level {
		t.Errorf("Level = %q, want %q", result.Level, obs.Level)
	}
	if result.SessionID != obs.SessionID {
		t.Errorf("SessionID = %q, want %q", result.SessionID, obs.SessionID)
	}
	if result.UserID != obs.UserID {
		t.Errorf("UserID = %q, want %q", result.UserID, obs.UserID)
	}
	if result.AppName != obs.AppName {
		t.Errorf("AppName = %q, want %q", result.AppName, obs.AppName)
	}
	if result.TimesDerived != obs.TimesDerived {
		t.Errorf("TimesDerived = %d, want %d", result.TimesDerived, obs.TimesDerived)
	}
	if len(result.Tags) != len(obs.Tags) || result.Tags[0] != obs.Tags[0] || result.Tags[1] != obs.Tags[1] {
		t.Errorf("Tags = %v, want %v", result.Tags, obs.Tags)
	}
	if len(result.Embedding) != len(obs.Embedding) {
		t.Errorf("Embedding length = %d, want %d", len(result.Embedding), len(obs.Embedding))
	}
}

func TestSQLiteStorage_StoreGetByID_NoTagsNoEmbedding(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	obs := &adapter.Observation{
		ID:        "obs-minimal",
		Content:   "minimal observation",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		CreatedAt: time.Now(),
	}

	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	result, err := storage.GetByID(ctx, "obs-minimal")
	if err != nil {
		t.Fatalf("GetByID() error = %v", err)
	}

	if result.Tags != nil {
		t.Errorf("Tags = %v, want nil", result.Tags)
	}
	if result.Embedding != nil {
		t.Errorf("Embedding = %v, want nil", result.Embedding)
	}
}

func TestSQLiteStorage_GetByID_NotFound(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	_, err = storage.GetByID(ctx, "nonexistent")
	if err == nil {
		t.Error("Expected error for nonexistent ID")
	}
}

func TestSQLiteStorage_Store_DuplicateID(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	obs := &adapter.Observation{
		ID:        "obs-dup",
		Content:   "first",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	// Store again with same ID
	dup := &adapter.Observation{
		ID:        "obs-dup",
		Content:   "second",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, dup); err == nil {
		t.Error("Expected error for duplicate ID")
	}
}

func TestSQLiteStorage_Search_DefaultModeIsHybrid(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	observations := []*adapter.Observation{
		{ID: "obs-1", Content: "Go is a programming language", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", Embedding: fakeEmbedding("golang programming"), CreatedAt: time.Now()},
		{ID: "obs-2", Content: "user enjoys hiking", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", Embedding: fakeEmbedding("outdoor hiking nature"), CreatedAt: time.Now()},
	}
	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	// Search with zero-value SearchMode and explicit MaxResults
	// Should use hybrid search (the default), not vector-fallback
	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "Go programming",
		MaxResults: 5,
		// Mode not set — zero value should be Hybrid
		SessionID: "s1",
		UserID:    "u1",
		AppName:   "a1",
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Expected results from default-mode search")
	}

	// Results should come from RRF fusion (hybrid), not vector-fallback
	foundRRF := false
	for _, r := range results {
		if r.Source == "rrf" {
			foundRRF = true
			break
		}
	}
	if !foundRRF {
		t.Error("Expected at least one result with source='rrf' from default hybrid search")
	}
}

func TestSQLiteStorage_SearchDoesNotMutateOpts(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	obs := &adapter.Observation{
		ID:        "obs-mutate",
		Content:   "test mutation",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	opts := &adapter.SearchOptions{
		Query:      "test",
		MaxResults: 0, // Should trigger default but NOT be mutated
		Mode:       adapter.SearchModeFTS,
	}

	_, err = storage.Search(ctx, opts)
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	if opts.MaxResults != 0 {
		t.Errorf("Search mutated opts.MaxResults: got %d, want 0", opts.MaxResults)
	}
}

func TestSQLiteStorage_Purge(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	// Store observations in two sessions
	for i, sessionID := range []string{"s1", "s2"} {
		obs := &adapter.Observation{
			ID:        "obs-purge-" + sessionID,
			Content:   "content for " + sessionID,
			Level:     adapter.LevelExplicit,
			SessionID: sessionID,
			UserID:    "u1",
			AppName:   "a1",
			CreatedAt: time.Now(),
		}
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store(%d) error = %v", i, err)
		}
	}

	// Purge session s1
	if err := storage.Purge(ctx, map[string]string{"session_id": "s1"}); err != nil {
		t.Fatalf("Purge() error = %v", err)
	}

	// s1 observation should be gone
	_, err = storage.GetByID(ctx, "obs-purge-s1")
	if err == nil {
		t.Error("Expected s1 observation to be purged")
	}

	// s2 observation should still exist
	result, err := storage.GetByID(ctx, "obs-purge-s2")
	if err != nil {
		t.Fatalf("GetByID(s2) error = %v", err)
	}
	if result.ID != "obs-purge-s2" {
		t.Errorf("s2 observation ID = %q, want 'obs-purge-s2'", result.ID)
	}
}

func TestSQLiteStorage_FTSSearch(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	observations := []*adapter.Observation{
		{ID: "obs-1", Content: "user enjoys hiking on weekends", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
		{ID: "obs-2", Content: "user works as a software engineer", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
		{ID: "obs-3", Content: "user likes pizza", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
	}

	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "hiking weekends",
		MaxResults: 5,
		Mode:       adapter.SearchModeFTS,
	})
	if err != nil {
		t.Fatalf("Search(FTS) error = %v", err)
	}

	if len(results) == 0 {
		t.Fatal("Expected FTS search results")
	}

	// obs-1 should be the best match
	found := false
	for _, r := range results {
		if r.Observation.ID == "obs-1" {
			found = true
			break
		}
	}
	if !found {
		t.Error("Expected obs-1 (hiking) in FTS results")
	}
}

func TestSQLiteStorage_FTSSearch_EmptyQuery(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	obs := &adapter.Observation{
		ID:        "obs-empty-q",
		Content:   "some content",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	// Empty query should fall back to recent observations
	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "",
		MaxResults: 5,
		Mode:       adapter.SearchModeFTS,
	})
	if err != nil {
		t.Fatalf("Search(FTS, empty query) error = %v", err)
	}
	if len(results) == 0 {
		t.Error("Expected fallback results for empty FTS query")
	}
}

func TestSQLiteStorage_Search_EmptyResults(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "nonexistent",
		MaxResults: 5,
		Mode:       adapter.SearchModeFTS,
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}
	if len(results) != 0 {
		t.Errorf("Expected 0 results for empty storage, got %d", len(results))
	}
}

func TestSQLiteStorage_IncrementTimesDerived_NotFound(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	// IncrementTimesDerived on nonexistent ID should return an error
	if err := storage.IncrementTimesDerived(ctx, "nonexistent"); err == nil {
		t.Error("Expected error for IncrementTimesDerived on nonexistent ID")
	}
}

func TestSQLiteStorage_Forget_NonexistentID(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	// Forget on nonexistent ID should return nil (idempotent delete)
	if err := storage.Forget(ctx, "nonexistent"); err != nil {
		t.Errorf("Forget(nonexistent) error = %v", err)
	}
}

func TestSQLiteStorage_Search_FilterBySessionID(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	observations := []*adapter.Observation{
		{ID: "obs-s1", Content: "session one fact", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
		{ID: "obs-s2", Content: "session two fact", Level: adapter.LevelExplicit, SessionID: "s2", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
	}

	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	// Search scoped to s1 should NOT return s2 observations
	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "fact",
		MaxResults: 10,
		Mode:       adapter.SearchModeFTS,
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

	// Verify s2 observation is not in results
	for _, r := range results {
		if r.Observation.ID == "obs-s2" {
			t.Error("obs-s2 should not appear in s1-scoped search")
		}
	}
}

func TestSQLiteStorage_Search_FilterByUserID(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	observations := []*adapter.Observation{
		{ID: "obs-u1", Content: "user one fact", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
		{ID: "obs-u2", Content: "user two fact", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u2", AppName: "a1", CreatedAt: time.Now()},
	}

	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "fact",
		MaxResults: 10,
		Mode:       adapter.SearchModeFTS,
		UserID:     "u1",
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	for _, r := range results {
		if r.Observation.UserID != "u1" {
			t.Errorf("Got result from user %s, want only u1", r.Observation.UserID)
		}
	}
}

func TestSQLiteStorage_Search_FilterByAppName(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	observations := []*adapter.Observation{
		{ID: "obs-a1", Content: "app one fact", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "app1", CreatedAt: time.Now()},
		{ID: "obs-a2", Content: "app two fact", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "app2", CreatedAt: time.Now()},
	}

	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "fact",
		MaxResults: 10,
		Mode:       adapter.SearchModeFTS,
		AppName:    "app1",
	})
	if err != nil {
		t.Fatalf("Search() error = %v", err)
	}

	for _, r := range results {
		if r.Observation.AppName != "app1" {
			t.Errorf("Got result from app %s, want only app1", r.Observation.AppName)
		}
	}
}

func TestSQLiteStorage_Purge_SyncsVirtualTables(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	obs := &adapter.Observation{
		ID:        "obs-purge-fts",
		Content:   "unique content for purge test",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		UserID:    "u1",
		AppName:   "a1",
		Embedding: fakeEmbedding("unique content for purge test"),
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	// Verify it exists in FTS
	ftsResults, _ := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "unique content purge",
		MaxResults: 1,
		Mode:       adapter.SearchModeFTS,
	})
	if len(ftsResults) == 0 {
		t.Fatal("Expected to find observation in FTS before purge")
	}

	// Purge by session_id
	if err := storage.Purge(ctx, map[string]string{"session_id": "s1"}); err != nil {
		t.Fatalf("Purge() error = %v", err)
	}

	// Verify it's gone from FTS
	ftsResults, _ = storage.Search(ctx, &adapter.SearchOptions{
		Query:      "unique content purge",
		MaxResults: 1,
		Mode:       adapter.SearchModeFTS,
	})
	for _, r := range ftsResults {
		if r.Observation.ID == "obs-purge-fts" {
			t.Error("Observation still found in FTS after purge")
		}
	}
}

func TestSQLiteStorage_Forget_SyncsVecTable(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	obs := &adapter.Observation{
		ID:        "obs-del-vec",
		Content:   "to be deleted from vec",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		UserID:    "u1",
		AppName:   "a1",
		Embedding: fakeEmbedding("test content for vec delete"),
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	// Verify it exists in vector search
	vecResults, err := storage.Search(ctx, &adapter.SearchOptions{
		Embedding:  fakeEmbedding("test content for vec delete"),
		MaxResults: 1,
		Mode:       adapter.SearchModeVector,
		SessionID:  "s1",
		UserID:     "u1",
		AppName:    "a1",
	})
	if err != nil {
		t.Fatalf("Vector search before delete error = %v", err)
	}
	if len(vecResults) == 0 {
		t.Fatal("Expected to find observation in vector search before deletion")
	}

	// Delete it
	if err := storage.Forget(ctx, "obs-del-vec"); err != nil {
		t.Fatalf("Forget() error = %v", err)
	}

	// Verify it's gone from main table
	_, err = storage.GetByID(ctx, "obs-del-vec")
	if err == nil {
		t.Error("Expected observation to be deleted from main table")
	}

	// Verify vector search no longer returns it
	vecResults, err = storage.Search(ctx, &adapter.SearchOptions{
		Embedding:  fakeEmbedding("test content for vec delete"),
		MaxResults: 10,
		Mode:       adapter.SearchModeVector,
		SessionID:  "s1",
		UserID:     "u1",
		AppName:    "a1",
	})
	if err != nil {
		t.Fatalf("Vector search after delete error = %v", err)
	}
	for _, r := range vecResults {
		if r.Observation.ID == "obs-del-vec" {
			t.Error("Observation still found in vector search after deletion")
		}
	}
}

func TestSQLiteStorage_Purge_UnknownFilterKey(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	obs := &adapter.Observation{
		ID:        "obs-unknown-filter",
		Content:   "test content",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		UserID:    "u1",
		AppName:   "a1",
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	// Purge with an unsupported filter key should return an error
	// to prevent accidental full deletion
	if err := storage.Purge(ctx, map[string]string{"level": "explicit"}); err == nil {
		t.Error("Expected error for Purge with unrecognized filter keys")
	}

	// adapter.Observation should still exist
	result, err := storage.GetByID(ctx, "obs-unknown-filter")
	if err != nil {
		t.Fatalf("GetByID() error = %v, observation should still exist", err)
	}
	if result.ID != "obs-unknown-filter" {
		t.Errorf("ID = %q, want 'obs-unknown-filter'", result.ID)
	}
}

func TestSQLiteStorage_Purge_MixedFilterKeys(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	obs := &adapter.Observation{
		ID:        "obs-mixed-filter",
		Content:   "test content",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		UserID:    "u1",
		AppName:   "a1",
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	// Purge with a mix of recognized and unrecognized filter keys should return an error
	// to prevent silent partial matching
	if err := storage.Purge(ctx, map[string]string{"session_id": "s1", "level": "explicit"}); err == nil {
		t.Error("Expected error for Purge with mixed recognized/unrecognized filter keys")
	}

	// adapter.Observation should still exist
	result, err := storage.GetByID(ctx, "obs-mixed-filter")
	if err != nil {
		t.Fatalf("GetByID() error = %v, observation should still exist", err)
	}
	if result.ID != "obs-mixed-filter" {
		t.Errorf("ID = %q, want 'obs-mixed-filter'", result.ID)
	}
}

func TestSQLiteStorage_Purge_EmptyFilter(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	obs := &adapter.Observation{
		ID:        "obs-empty-filter",
		Content:   "test content",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		UserID:    "u1",
		AppName:   "a1",
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	// Purge with empty filter should return an error
	if err := storage.Purge(ctx, map[string]string{}); err == nil {
		t.Error("Expected error for Purge with empty filter")
	}

	// adapter.Observation should still exist
	result, err := storage.GetByID(ctx, "obs-empty-filter")
	if err != nil {
		t.Fatalf("GetByID() error = %v, observation should still exist", err)
	}
	if result.ID != "obs-empty-filter" {
		t.Errorf("ID = %q, want 'obs-empty-filter'", result.ID)
	}
}

func TestSQLiteStorage_Purge_NoMatchingRows(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	obs := &adapter.Observation{
		ID:        "obs-no-match",
		Content:   "test content",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		UserID:    "u1",
		AppName:   "a1",
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	// Purge with a valid filter that matches nothing should succeed silently
	if err := storage.Purge(ctx, map[string]string{"session_id": "nonexistent"}); err != nil {
		t.Errorf("Purge with no matching rows should not error, got: %v", err)
	}

	// Original observation should still exist (different session_id)
	result, err := storage.GetByID(ctx, "obs-no-match")
	if err != nil {
		t.Fatalf("GetByID() error = %v, observation should still exist", err)
	}
	if result.ID != "obs-no-match" {
		t.Errorf("ID = %q, want 'obs-no-match'", result.ID)
	}
}
