package sqlite

import (
	"context"
	"testing"
	"time"

	"github.com/ieshan/adk-go-memory/adapter"
	"github.com/ieshan/adk-go-pkg/testutil"
	"github.com/ieshan/idx"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
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
	idGo := idx.NewID()
	idPizza := idx.NewID()
	idPy := idx.NewID()
	observations := []*adapter.Observation{
		{
			ID:        idGo,
			Content:   "user likes Go programming",
			Level:     adapter.LevelExplicit,
			SessionID: "s1",
			UserID:    "u1",
			AppName:   "a1",
			Embedding: testutil.FakeEmbed("golang programming language"),
			CreatedAt: time.Now(),
		},
		{
			ID:        idPizza,
			Content:   "user enjoys pizza",
			Level:     adapter.LevelExplicit,
			SessionID: "s1",
			UserID:    "u1",
			AppName:   "a1",
			Embedding: testutil.FakeEmbed("italian food pizza cheese"),
			CreatedAt: time.Now(),
		},
		{
			ID:        idPy,
			Content:   "user knows Python",
			Level:     adapter.LevelExplicit,
			SessionID: "s1",
			UserID:    "u1",
			AppName:   "a1",
			Embedding: testutil.FakeEmbed("python programming code"),
			CreatedAt: time.Now(),
		},
	}

	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	// Search with embedding similar to Go/programming
	queryEmbedding := testutil.FakeEmbed("golang development")
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
		if r.Observation.ID == idGo || r.Observation.ID == idPy {
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
	idGoText := idx.NewID()
	idGoVec := idx.NewID()
	observations := []*adapter.Observation{
		{
			ID:           idGoText,
			Content:      "Go is a programming language created at Google",
			Level:        adapter.LevelExplicit,
			SessionID:    "s1",
			UserID:       "u1",
			AppName:      "a1",
			Embedding:    testutil.FakeEmbed("completely unrelated text about food"),
			TimesDerived: 1,
			CreatedAt:    time.Now().Add(-1 * time.Hour),
		},
		{
			ID:           idGoVec,
			Content:      "The user mentioned they use Golang for backend work",
			Level:        adapter.LevelExplicit,
			SessionID:    "s1",
			UserID:       "u1",
			AppName:      "a1",
			Embedding:    testutil.FakeEmbed("golang programming backend development"),
			TimesDerived: 5, // Highly derived
			CreatedAt:    time.Now(),
		},
	}

	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Store() error = %v", err)
		}
	}

	queryEmbedding := testutil.FakeEmbed("golang programming")
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
	ids := make(map[idx.ID]bool)
	rrfCount := 0
	for _, r := range results {
		ids[r.Observation.ID] = true
		if r.Source == "rrf" {
			rrfCount++
		}
	}

	if !ids[idGoText] {
		t.Error("Expected obs-go-text in hybrid results (FTS match)")
	}
	if !ids[idGoVec] {
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

	idLow := idx.NewID()
	idHigh := idx.NewID()
	idMedium := idx.NewID()
	observations := []*adapter.Observation{
		{ID: idLow, Content: "low derived", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 1, CreatedAt: time.Now()},
		{ID: idHigh, Content: "high derived", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 10, CreatedAt: time.Now().Add(-1 * time.Hour)},
		{ID: idMedium, Content: "medium derived", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", TimesDerived: 5, CreatedAt: time.Now().Add(-2 * time.Hour)},
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
	if results[0].ID != idHigh || results[0].TimesDerived != 10 {
		t.Errorf("Expected high-derived (TimesDerived=10) first, got %s (TimesDerived=%d)", results[0].ID.String(), results[0].TimesDerived)
	}
	if results[1].ID != idMedium || results[1].TimesDerived != 5 {
		t.Errorf("Expected medium-derived (TimesDerived=5) second, got %s (TimesDerived=%d)", results[1].ID.String(), results[1].TimesDerived)
	}
	if results[2].ID != idLow || results[2].TimesDerived != 1 {
		t.Errorf("Expected low-derived (TimesDerived=1) third, got %s (TimesDerived=%d)", results[2].ID.String(), results[2].TimesDerived)
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
	// IDs are ULID-based and time-ordered; create them in chronological order
	// so that id DESC corresponds to recency.
	idOld := idx.NewID()
	idMiddle := idx.NewID()
	idNew := idx.NewID()
	observations := []*adapter.Observation{
		{ID: idOld, Content: "old", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: now.Add(-2 * time.Hour)},
		{ID: idMiddle, Content: "middle", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: now.Add(-1 * time.Hour)},
		{ID: idNew, Content: "new", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: now},
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

	// Should be ordered by id DESC (ULID-based, time-ordered)
	if results[0].ID != idNew {
		t.Errorf("Expected new observation first (newest id), got %s", results[0].ID.String())
	}
	if results[1].ID != idMiddle {
		t.Errorf("Expected middle observation second, got %s", results[1].ID.String())
	}
	if results[2].ID != idOld {
		t.Errorf("Expected old observation third (oldest id), got %s", results[2].ID.String())
	}
}

func TestSQLiteStorage_Forget_SyncsVirtualTables(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	idDelete := idx.NewID()
	obs := &adapter.Observation{
		ID:        idDelete,
		Content:   "to be deleted",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		UserID:    "u1",
		AppName:   "a1",
		Embedding: testutil.FakeEmbed("test content"),
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
	if err := storage.Forget(ctx, idDelete); err != nil {
		t.Fatalf("Forget() error = %v", err)
	}

	// Verify it's gone from main table
	_, err = storage.GetByID(ctx, idDelete)
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
		if r.Observation.ID == idDelete {
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

	idIncrement := idx.NewID()
	obs := &adapter.Observation{
		ID:           idIncrement,
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
	if err := storage.IncrementTimesDerived(ctx, idIncrement); err != nil {
		t.Fatalf("IncrementTimesDerived error = %v", err)
	}

	// Verify
	result, err := storage.GetByID(ctx, idIncrement)
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
	idFull := idx.NewID()
	obs := &adapter.Observation{
		ID:           idFull,
		Content:      "full round trip test",
		Level:        adapter.LevelDeductive,
		SessionID:    "s1",
		UserID:       "u1",
		AppName:      "a1",
		Tags:         []string{"tag1", "tag2"},
		TimesDerived: 5,
		CreatedAt:    now,
		Embedding:    testutil.FakeEmbed("test embedding round trip"),
	}

	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	result, err := storage.GetByID(ctx, idFull)
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

	idMinimal := idx.NewID()
	obs := &adapter.Observation{
		ID:        idMinimal,
		Content:   "minimal observation",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		CreatedAt: time.Now(),
	}

	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	result, err := storage.GetByID(ctx, idMinimal)
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

	_, err = storage.GetByID(ctx, idx.NewID())
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

	idDup := idx.NewID()
	obs := &adapter.Observation{
		ID:        idDup,
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
		ID:        idDup,
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
		{ID: idx.NewID(), Content: "Go is a programming language", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", Embedding: testutil.FakeEmbed("golang programming"), CreatedAt: time.Now()},
		{ID: idx.NewID(), Content: "user enjoys hiking", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", Embedding: testutil.FakeEmbed("outdoor hiking nature"), CreatedAt: time.Now()},
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
		ID:        idx.NewID(),
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
	ids := make(map[string]idx.ID)
	for i, sessionID := range []string{"s1", "s2"} {
		id := idx.NewID()
		ids[sessionID] = id
		obs := &adapter.Observation{
			ID:        id,
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
	_, err = storage.GetByID(ctx, ids["s1"])
	if err == nil {
		t.Error("Expected s1 observation to be purged")
	}

	// s2 observation should still exist
	result, err := storage.GetByID(ctx, ids["s2"])
	if err != nil {
		t.Fatalf("GetByID(s2) error = %v", err)
	}
	if result.ID != ids["s2"] {
		t.Errorf("s2 observation ID = %q, want %q", result.ID.String(), ids["s2"].String())
	}
}

func TestSQLiteStorage_FTSSearch(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	idObs1 := idx.NewID()
	observations := []*adapter.Observation{
		{ID: idObs1, Content: "user enjoys hiking on weekends", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
		{ID: idx.NewID(), Content: "user works as a software engineer", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
		{ID: idx.NewID(), Content: "user likes pizza", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
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
		if r.Observation.ID == idObs1 {
			found = true
			break
		}
	}
	if !found {
		t.Error("Expected hiking observation in FTS results")
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
		ID:        idx.NewID(),
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
	if err := storage.IncrementTimesDerived(ctx, idx.NewID()); err == nil {
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
	if err := storage.Forget(ctx, idx.NewID()); err != nil {
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

	idS2 := idx.NewID()
	observations := []*adapter.Observation{
		{ID: idx.NewID(), Content: "session one fact", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
		{ID: idS2, Content: "session two fact", Level: adapter.LevelExplicit, SessionID: "s2", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
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
		if r.Observation.ID == idS2 {
			t.Error("s2 observation should not appear in s1-scoped search")
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
		{ID: idx.NewID(), Content: "user one fact", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "a1", CreatedAt: time.Now()},
		{ID: idx.NewID(), Content: "user two fact", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u2", AppName: "a1", CreatedAt: time.Now()},
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
		{ID: idx.NewID(), Content: "app one fact", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "app1", CreatedAt: time.Now()},
		{ID: idx.NewID(), Content: "app two fact", Level: adapter.LevelExplicit, SessionID: "s1", UserID: "u1", AppName: "app2", CreatedAt: time.Now()},
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

	idPurge := idx.NewID()
	obs := &adapter.Observation{
		ID:        idPurge,
		Content:   "unique content for purge test",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		UserID:    "u1",
		AppName:   "a1",
		Embedding: testutil.FakeEmbed("unique content for purge test"),
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
		if r.Observation.ID == idPurge {
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

	idDelVec := idx.NewID()
	obs := &adapter.Observation{
		ID:        idDelVec,
		Content:   "to be deleted from vec",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		UserID:    "u1",
		AppName:   "a1",
		Embedding: testutil.FakeEmbed("test content for vec delete"),
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	// Verify it exists in vector search
	vecResults, err := storage.Search(ctx, &adapter.SearchOptions{
		Embedding:  testutil.FakeEmbed("test content for vec delete"),
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
	if err := storage.Forget(ctx, idDelVec); err != nil {
		t.Fatalf("Forget() error = %v", err)
	}

	// Verify it's gone from main table
	_, err = storage.GetByID(ctx, idDelVec)
	if err == nil {
		t.Error("Expected observation to be deleted from main table")
	}

	// Verify vector search no longer returns it
	vecResults, err = storage.Search(ctx, &adapter.SearchOptions{
		Embedding:  testutil.FakeEmbed("test content for vec delete"),
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
		if r.Observation.ID == idDelVec {
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

	idUnknown := idx.NewID()
	obs := &adapter.Observation{
		ID:        idUnknown,
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
	result, err := storage.GetByID(ctx, idUnknown)
	if err != nil {
		t.Fatalf("GetByID() error = %v, observation should still exist", err)
	}
	if result.ID != idUnknown {
		t.Errorf("ID = %q, want %q", result.ID.String(), idUnknown.String())
	}
}

func TestSQLiteStorage_Purge_MixedFilterKeys(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	idMixed := idx.NewID()
	obs := &adapter.Observation{
		ID:        idMixed,
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
	result, err := storage.GetByID(ctx, idMixed)
	if err != nil {
		t.Fatalf("GetByID() error = %v, observation should still exist", err)
	}
	if result.ID != idMixed {
		t.Errorf("ID = %q, want %q", result.ID.String(), idMixed.String())
	}
}

func TestNewSQLiteStorageWithGORM(t *testing.T) {
	ctx := context.Background()

	// Create a GORM database connection
	db, err := gorm.Open(sqlite.Open("file::memory:?cache=shared"), &gorm.Config{})
	if err != nil {
		t.Fatalf("Failed to open database: %v", err)
	}

	// Create storage from existing connection
	storage, err := NewSQLiteStorageWithGORM(db)
	if err != nil {
		t.Fatalf("NewSQLiteStorageWithGORM() error = %v", err)
	}

	// Verify storage works
	id := idx.NewID()
	obs := &adapter.Observation{
		ID:        id,
		Content:   "test from existing db",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		UserID:    "u1",
		AppName:   "a1",
		CreatedAt: time.Now(),
	}

	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store() error = %v", err)
	}

	result, err := storage.GetByID(ctx, id)
	if err != nil {
		t.Fatalf("GetByID() error = %v", err)
	}

	if result.Content != obs.Content {
		t.Errorf("Content = %q, want %q", result.Content, obs.Content)
	}

	// Close storage - should NOT close the underlying db
	if err := storage.Close(); err != nil {
		t.Fatalf("storage.Close() error = %v", err)
	}

	// Verify db is still usable (not closed)
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("db.DB() error = %v", err)
	}
	if err := sqlDB.Ping(); err != nil {
		t.Errorf("db.Ping() failed after storage.Close(): %v", err)
	}
	sqlDB.Close()
}

func TestNewSQLiteStorageWithGORM_Migrations(t *testing.T) {
	ctx := context.Background()

	// Create a GORM database connection
	db, err := gorm.Open(sqlite.Open("file::memory:?cache=shared"), &gorm.Config{})
	if err != nil {
		t.Fatalf("Failed to open database: %v", err)
	}

	// Create storage - should apply migrations
	storage, err := NewSQLiteStorageWithGORM(db)
	if err != nil {
		t.Fatalf("NewSQLiteStorageWithGORM() error = %v", err)
	}

	// Verify schema was created by checking a table exists
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("db.DB() error = %v", err)
	}
	defer sqlDB.Close()

	var count int
	err = sqlDB.QueryRowContext(ctx,
		"SELECT count(*) FROM sqlite_master WHERE type='table' AND name='observations'").Scan(&count)
	if err != nil {
		t.Fatalf("Failed to check for observations table: %v", err)
	}
	if count != 1 {
		t.Errorf("observations table not found, count = %d", count)
	}

	// Verify FTS table exists
	err = sqlDB.QueryRowContext(ctx,
		"SELECT count(*) FROM sqlite_master WHERE type='table' AND name='observations_fts'").Scan(&count)
	if err != nil {
		t.Fatalf("Failed to check for observations_fts table: %v", err)
	}
	if count != 1 {
		t.Errorf("observations_fts table not found, count = %d", count)
	}

	// Verify vec table exists
	err = sqlDB.QueryRowContext(ctx,
		"SELECT count(*) FROM sqlite_master WHERE type='table' AND name='vec_observations'").Scan(&count)
	if err != nil {
		t.Fatalf("Failed to check for vec_observations table: %v", err)
	}
	if count != 1 {
		t.Errorf("vec_observations table not found, count = %d", count)
	}

	storage.Close()
}

func TestNewSQLiteStorageWithGORM_NilDB(t *testing.T) {
	// Test with nil database
	_, err := NewSQLiteStorageWithGORM(nil)
	if err == nil {
		t.Error("Expected error for nil database")
	}
}

func TestNewSQLiteStorageWithGORM_ReuseConnection(t *testing.T) {
	ctx := context.Background()

	// Create a shared GORM database connection
	db, err := gorm.Open(sqlite.Open("file::memory:?cache=shared"), &gorm.Config{})
	if err != nil {
		t.Fatalf("Failed to open database: %v", err)
	}

	// Create first storage instance
	storage1, err := NewSQLiteStorageWithGORM(db)
	if err != nil {
		t.Fatalf("NewSQLiteStorageWithGORM() #1 error = %v", err)
	}

	// Store something
	id1 := idx.NewID()
	obs1 := &adapter.Observation{
		ID:        id1,
		Content:   "first storage",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		UserID:    "u1",
		AppName:   "a1",
		CreatedAt: time.Now(),
	}
	if err := storage1.Store(ctx, obs1); err != nil {
		t.Fatalf("storage1.Store() error = %v", err)
	}
	storage1.Close()

	// Create second storage instance with same connection
	storage2, err := NewSQLiteStorageWithGORM(db)
	if err != nil {
		t.Fatalf("NewSQLiteStorageWithGORM() #2 error = %v", err)
	}

	// Verify we can read what storage1 wrote
	result, err := storage2.GetByID(ctx, id1)
	if err != nil {
		t.Fatalf("storage2.GetByID() error = %v", err)
	}
	if result.Content != "first storage" {
		t.Errorf("Content = %q, want %q", result.Content, "first storage")
	}

	// Store something new
	id2 := idx.NewID()
	obs2 := &adapter.Observation{
		ID:        id2,
		Content:   "second storage",
		Level:     adapter.LevelExplicit,
		SessionID: "s1",
		UserID:    "u1",
		AppName:   "a1",
		CreatedAt: time.Now(),
	}
	if err := storage2.Store(ctx, obs2); err != nil {
		t.Fatalf("storage2.Store() error = %v", err)
	}
	storage2.Close()

	// Verify data is still accessible through raw db
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("db.DB() error = %v", err)
	}
	var content string
	err = sqlDB.QueryRowContext(ctx,
		"SELECT content FROM observations WHERE id = ?", id2).Scan(&content)
	if err != nil {
		t.Fatalf("Query through raw db failed: %v", err)
	}
	if content != "second storage" {
		t.Errorf("Content from raw db = %q, want %q", content, "second storage")
	}
	sqlDB.Close()
}

func TestSQLiteStorage_Purge_EmptyFilter(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	idEmpty := idx.NewID()
	obs := &adapter.Observation{
		ID:        idEmpty,
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
	result, err := storage.GetByID(ctx, idEmpty)
	if err != nil {
		t.Fatalf("GetByID() error = %v, observation should still exist", err)
	}
	if result.ID != idEmpty {
		t.Errorf("ID = %q, want %q", result.ID.String(), idEmpty.String())
	}
}

func TestSQLiteStorage_Purge_NoMatchingRows(t *testing.T) {
	ctx := context.Background()
	storage, err := InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	idNoMatch := idx.NewID()
	obs := &adapter.Observation{
		ID:        idNoMatch,
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
	result, err := storage.GetByID(ctx, idNoMatch)
	if err != nil {
		t.Fatalf("GetByID() error = %v, observation should still exist", err)
	}
	if result.ID != idNoMatch {
		t.Errorf("ID = %q, want %q", result.ID.String(), idNoMatch.String())
	}
}
