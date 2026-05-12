package compaction

import (
	"context"
	"testing"
	"time"

	"github.com/ieshan/adk-go-memory/adapter"
	"github.com/ieshan/adk-go-pkg/testutil"
	"github.com/ieshan/idx"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestNewBackgroundCompactor(t *testing.T) {
	storage := adapter.InMemory()
	defer storage.Close()

	llm := testutil.NewFakeLLM()
	bc := NewBackgroundCompactor(storage, llm)

	if bc == nil {
		t.Fatal("NewBackgroundCompactor() returned nil")
	}
	if bc.storage != storage {
		t.Error("BackgroundCompactor storage not set correctly")
	}
	if bc.llm != llm {
		t.Error("BackgroundCompactor llm not set correctly")
	}
}

func TestBackgroundCompactor_QueryObservationsForCompaction(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	// Create test observations with different ages
	now := time.Now()
	id1, id2, id3 := idx.NewID(), idx.NewID(), idx.NewID()
	observations := []*adapter.Observation{
		{
			ID:        id1,
			Content:   "old observation from last year",
			Level:     adapter.LevelExplicit,
			SessionID: "sess-1",
			UserID:    "user1",
			AppName:   "app1",
			CreatedAt: now.Add(-365 * 24 * time.Hour), // 1 year ago
		},
		{
			ID:        id2,
			Content:   "recent observation from today",
			Level:     adapter.LevelExplicit,
			SessionID: "sess-1",
			UserID:    "user1",
			AppName:   "app1",
			CreatedAt: now,
		},
		{
			ID:        id3,
			Content:   "medium age observation from last month",
			Level:     adapter.LevelExplicit,
			SessionID: "sess-1",
			UserID:    "user1",
			AppName:   "app1",
			CreatedAt: now.Add(-30 * 24 * time.Hour), // 30 days ago
		},
	}

	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Failed to store observation: %v", err)
		}
	}

	bc := NewBackgroundCompactor(storage, nil)

	tests := []struct {
		name        string
		opts        CompactionQueryOptions
		wantCount   int
		wantMinAge  time.Duration
		wantContent string
	}{
		{
			name: "query all observations",
			opts: CompactionQueryOptions{
				MaxResults: 10,
			},
			wantCount: 3,
		},
		{
			name: "query with OlderThan filter",
			opts: CompactionQueryOptions{
				OlderThan:  now.Add(-7 * 24 * time.Hour), // Older than 7 days
				MaxResults: 10,
			},
			wantCount: 2, // obs-1 and obs-3
		},
		{
			name: "query with MinAge filter",
			opts: CompactionQueryOptions{
				MinAge:     25 * 24 * time.Hour, // At least 25 days old
				MaxResults: 10,
			},
			wantCount: 2, // obs-1 and obs-3
		},
		{
			name: "query with MaxResults limit",
			opts: CompactionQueryOptions{
				MaxResults: 1,
			},
			wantCount: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := bc.QueryObservationsForCompaction(ctx, tt.opts)
			if err != nil {
				t.Fatalf("QueryObservationsForCompaction() error = %v", err)
			}
			if len(results) != tt.wantCount {
				t.Errorf("QueryObservationsForCompaction() returned %d observations, want %d", len(results), tt.wantCount)
			}
		})
	}
}

func TestBackgroundCompactor_CreateCompactionSummary(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	llm := testutil.NewFakeLLM(testutil.NewTextResponse("Summary of old observations about user preferences and facts"))
	bc := NewBackgroundCompactor(storage, llm)

	now := time.Now()
	id1, id2 := idx.NewID(), idx.NewID()
	observations := []CompactionObservation{
		{
			Observation: adapter.Observation{
				ID:        id1,
				Content:   "user likes Go programming",
				Level:     adapter.LevelExplicit,
				UserID:    "user1",
				AppName:   "app1",
				SessionID: "sess-1",
				CreatedAt: now.Add(-100 * 24 * time.Hour),
			},
			Age: 100 * 24 * time.Hour,
		},
		{
			Observation: adapter.Observation{
				ID:        id2,
				Content:   "user prefers coffee over tea",
				Level:     adapter.LevelDeductive,
				UserID:    "user1",
				AppName:   "app1",
				SessionID: "sess-1",
				CreatedAt: now.Add(-90 * 24 * time.Hour),
			},
			Age: 90 * 24 * time.Hour,
		},
	}

	// Test successful summary creation
	opts := SummaryOptions{
		Instruction: "Summarize these user observations",
		MaxTokens:   500,
		Tags:        []string{"user_preferences"},
	}

	summary, err := bc.CreateCompactionSummary(ctx, observations, opts)
	if err != nil {
		t.Fatalf("CreateCompactionSummary() error = %v", err)
	}

	if summary == nil {
		t.Fatal("CreateCompactionSummary() returned nil")
	}
	if summary.Content != "Summary of old observations about user preferences and facts" {
		t.Errorf("Summary content = %q, want mock response", summary.Content)
	}
	if summary.UserID != "user1" {
		t.Errorf("Summary UserID = %q, want user1", summary.UserID)
	}
	if summary.AppName != "app1" {
		t.Errorf("Summary AppName = %q, want app1", summary.AppName)
	}
	if summary.SessionID != "sess-1" {
		t.Errorf("Summary SessionID = %q, want sess-1", summary.SessionID)
	}
	if summary.Level != adapter.LevelInductive {
		t.Errorf("Summary Level = %q, want inductive", summary.Level)
	}

	// Check compaction_summary tag is added
	hasCompactionTag := false
	for _, tag := range summary.Tags {
		if tag == "compaction_summary" {
			hasCompactionTag = true
			break
		}
	}
	if !hasCompactionTag {
		t.Error("Summary should have 'compaction_summary' tag")
	}

	// Check custom tags are added
	hasCustomTag := false
	for _, tag := range summary.Tags {
		if tag == "user_preferences" {
			hasCustomTag = true
			break
		}
	}
	if !hasCustomTag {
		t.Error("Summary should have custom 'user_preferences' tag")
	}
}

func TestBackgroundCompactor_CreateCompactionSummary_NoLLM(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	// Create compactor without LLM
	bc := NewBackgroundCompactor(storage, nil)

	observations := []CompactionObservation{
		{
			Observation: adapter.Observation{
				ID:      idx.NewID(),
				Content: "test observation",
			},
		},
	}

	_, err := bc.CreateCompactionSummary(ctx, observations, SummaryOptions{})
	if err == nil {
		t.Error("CreateCompactionSummary() expected error when LLM is nil")
	}
}

func TestBackgroundCompactor_CreateCompactionSummary_EmptyObservations(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	llm := testutil.NewFakeLLM()
	bc := NewBackgroundCompactor(storage, llm)

	_, err := bc.CreateCompactionSummary(ctx, []CompactionObservation{}, SummaryOptions{})
	if err == nil {
		t.Error("CreateCompactionSummary() expected error for empty observations")
	}
}

func TestBackgroundCompactor_CreateCompactionSummary_EmptySummaryResponse(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	// Mock LLM that returns empty response
	llm := testutil.NewFakeLLM(testutil.NewTextResponse(""))
	bc := NewBackgroundCompactor(storage, llm)

	observations := []CompactionObservation{
		{
			Observation: adapter.Observation{
				ID:      idx.NewID(),
				Content: "test observation",
			},
		},
	}

	_, err := bc.CreateCompactionSummary(ctx, observations, SummaryOptions{})
	if err == nil {
		t.Error("CreateCompactionSummary() expected error for empty summary")
	}
}

func TestBackgroundCompactor_CreateCompactionSummary_DefaultPrompt(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	llm := testutil.NewFakeLLM(testutil.NewTextResponse("Default prompt summary"))
	bc := NewBackgroundCompactor(storage, llm)

	observations := []CompactionObservation{
		{
			Observation: adapter.Observation{
				ID:      idx.NewID(),
				Content: "test observation",
				Level:   adapter.LevelExplicit,
			},
		},
	}

	// Test with empty instruction (should use default)
	opts := SummaryOptions{
		Instruction: "", // Empty, should use default
	}

	summary, err := bc.CreateCompactionSummary(ctx, observations, opts)
	if err != nil {
		t.Fatalf("CreateCompactionSummary() with default prompt error = %v", err)
	}
	if summary.Content != "Default prompt summary" {
		t.Errorf("Summary content = %q, want 'Default prompt summary'", summary.Content)
	}
}

func TestBackgroundCompactor_ArchiveObservations(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	// Create test observations
	id1, id2 := idx.NewID(), idx.NewID()
	observations := []*adapter.Observation{
		{
			ID:        id1,
			Content:   "fact to archive",
			Level:     adapter.LevelExplicit,
			UserID:    "user1",
			AppName:   "app1",
			CreatedAt: time.Now(),
			Tags:      []string{"original"},
		},
		{
			ID:        id2,
			Content:   "another fact to archive",
			Level:     adapter.LevelExplicit,
			UserID:    "user1",
			AppName:   "app1",
			CreatedAt: time.Now(),
			Tags:      []string{"original"},
		},
	}

	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Failed to store observation: %v", err)
		}
	}

	bc := NewBackgroundCompactor(storage, nil)

	// Archive observations
	ids := []idx.ID{id1, id2}
	if err := bc.ArchiveObservations(ctx, ids); err != nil {
		t.Fatalf("ArchiveObservations() error = %v", err)
	}

	// Verify archived tag was added
	for _, id := range ids {
		obs, err := storage.GetByID(ctx, id)
		if err != nil {
			t.Fatalf("Failed to get archived observation: %v", err)
		}

		hasArchivedTag := false
		for _, tag := range obs.Tags {
			if tag == "archived" {
				hasArchivedTag = true
				break
			}
		}
		if !hasArchivedTag {
			t.Errorf("Observation %s should have 'archived' tag, got tags: %v", id, obs.Tags)
		}

		// Verify original tag is preserved
		hasOriginalTag := false
		for _, tag := range obs.Tags {
			if tag == "original" {
				hasOriginalTag = true
				break
			}
		}
		if !hasOriginalTag {
			t.Errorf("Observation %s should preserve 'original' tag, got tags: %v", id, obs.Tags)
		}
	}
}

func TestBackgroundCompactor_ArchiveObservations_NonExistent(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	bc := NewBackgroundCompactor(storage, nil)

	// Try to archive non-existent observation
	err := bc.ArchiveObservations(ctx, []idx.ID{idx.NewID()})
	if err == nil {
		t.Error("ArchiveObservations() expected error for non-existent observation")
	}
}

func TestBackgroundCompactor_PurgeArchivedObservations(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	now := time.Now()
	// Create observations with and without archived tag
	idOld, idRecent, idNotArchived := idx.NewID(), idx.NewID(), idx.NewID()
	observations := []*adapter.Observation{
		{
			ID:        idOld,
			Content:   "old archived observation",
			Level:     adapter.LevelExplicit,
			UserID:    "user1",
			AppName:   "app1",
			CreatedAt: now.Add(-100 * 24 * time.Hour), // 100 days ago
			Tags:      []string{"archived"},
		},
		{
			ID:        idRecent,
			Content:   "recent archived observation",
			Level:     adapter.LevelExplicit,
			UserID:    "user1",
			AppName:   "app1",
			CreatedAt: now.Add(-1 * 24 * time.Hour), // 1 day ago
			Tags:      []string{"archived"},
		},
		{
			ID:        idNotArchived,
			Content:   "not archived observation",
			Level:     adapter.LevelExplicit,
			UserID:    "user1",
			AppName:   "app1",
			CreatedAt: now.Add(-100 * 24 * time.Hour),
			Tags:      []string{},
		},
	}

	for _, obs := range observations {
		if err := storage.Store(ctx, obs); err != nil {
			t.Fatalf("Failed to store observation: %v", err)
		}
	}

	bc := NewBackgroundCompactor(storage, nil)

	// Purge observations older than 7 days
	count, err := bc.PurgeArchivedObservations(ctx, now.Add(-7*24*time.Hour))
	if err != nil {
		t.Fatalf("PurgeArchivedObservations() error = %v", err)
	}

	// Should remove obs-archived-old (100 days old)
	if count != 1 {
		t.Errorf("PurgeArchivedObservations() removed %d observations, want 1", count)
	}

	// Verify old archived observation was removed
	_, err = storage.GetByID(ctx, idOld)
	if err == nil {
		t.Error("Old archived observation should have been purged")
	}

	// Verify recent archived observation still exists
	_, err = storage.GetByID(ctx, idRecent)
	if err != nil {
		t.Error("Recent archived observation should still exist")
	}

	// Verify non-archived old observation still exists (not purged because not archived)
	_, err = storage.GetByID(ctx, idNotArchived)
	if err != nil {
		t.Error("Non-archived observation should still exist")
	}
}

func TestBackgroundCompactor_PurgeArchivedObservations_EmptyStorage(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	bc := NewBackgroundCompactor(storage, nil)

	count, err := bc.PurgeArchivedObservations(ctx, time.Now().Add(-7*24*time.Hour))
	if err != nil {
		t.Fatalf("PurgeArchivedObservations() on empty storage error = %v", err)
	}
	if count != 0 {
		t.Errorf("PurgeArchivedObservations() on empty storage removed %d, want 0", count)
	}
}

// Mock LLM for testing

func TestExtractSummaryText(t *testing.T) {
	tests := []struct {
		name     string
		response *model.LLMResponse
		want     string
	}{
		{
			name:     "nil response",
			response: nil,
			want:     "",
		},
		{
			name:     "nil content",
			response: &model.LLMResponse{Content: nil},
			want:     "",
		},
		{
			name:     "valid response with text",
			response: &model.LLMResponse{Content: genai.NewContentFromText("Summary text", genai.RoleModel)},
			want:     "Summary text",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ExtractSummaryText(tt.response)
			if got != tt.want {
				t.Errorf("ExtractSummaryText() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestBackgroundCompactor_CreateCompactionSummary_ContextCancelled(t *testing.T) {
	storage := adapter.InMemory()
	llm := testutil.NewFakeLLM(testutil.NewTextResponse("summary text"))
	bc := NewBackgroundCompactor(storage, llm)

	observations := []CompactionObservation{
		{Observation: adapter.Observation{ID: idx.NewID(), Content: "test content"}},
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := bc.CreateCompactionSummary(ctx, observations, SummaryOptions{})
	if err == nil {
		t.Error("Expected error for cancelled context, got nil")
	}
}
