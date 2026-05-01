package memory

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/ieshan/adk-go-memory/adapter"
	"github.com/ieshan/adk-go-pkg/testutil"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestSummarizer_ShouldSummarize(t *testing.T) {
	cfg := SummarizerConfig{ShortInterval: 20, LongInterval: 60}
	s := NewSummarizer(cfg)

	tests := []struct {
		count    int
		wantType SummaryType
		wantBool bool
	}{
		{0, "", false},
		{1, "", false},
		{19, "", false},
		{20, SummaryTypeShort, true},
		{40, SummaryTypeShort, true}, // 40 is multiple of 20 but not 60
		{60, SummaryTypeLong, true},  // 60 is multiple of both, long takes priority
		{80, SummaryTypeShort, true},
		{100, SummaryTypeShort, true},
		{120, SummaryTypeLong, true}, // 120 is multiple of both, long takes priority
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("count_%d", tt.count), func(t *testing.T) {
			gotType, gotBool := s.ShouldSummarize(tt.count)
			if gotBool != tt.wantBool {
				t.Errorf("ShouldSummarize(%d) bool = %v, want %v", tt.count, gotBool, tt.wantBool)
			}
			if gotType != tt.wantType {
				t.Errorf("ShouldSummarize(%d) type = %q, want %q", tt.count, gotType, tt.wantType)
			}
		})
	}
}

func TestSummarizer_Summarize(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	llm := testutil.NewFakeLLM(model.LLMResponse{
		Content: &genai.Content{
			Parts: []*genai.Part{{
				Text: "Summary of observations about Go programming.",
			}},
		},
	})

	summarizer := NewSummarizer(SummarizerConfig{
		LLM:     llm,
		Storage: storage,
	})

	msgs := []TimestampedMessage{
		{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "I love Go"}}}, At: time.Now()},
	}

	summary, err := summarizer.Summarize(ctx, msgs)
	if err != nil {
		t.Fatalf("Summarize() error = %v", err)
	}

	if summary.Content == "" {
		t.Error("Expected non-empty summary content")
	}

	if summary.CreatedAt.IsZero() {
		t.Error("Expected CreatedAt to be set")
	}
}

func TestSummarizer_Summarize_EmptyMessages(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	llm := testutil.NewFakeLLM()
	summarizer := NewSummarizer(SummarizerConfig{
		LLM:     llm,
		Storage: storage,
	})

	_, err := summarizer.Summarize(ctx, []TimestampedMessage{})
	if err == nil {
		t.Error("Expected error for empty messages")
	}
}

func TestSummarizer_DefaultIntervals(t *testing.T) {
	storage := adapter.InMemory()
	defer storage.Close()

	llm := testutil.NewFakeLLM()
	summarizer := NewSummarizer(SummarizerConfig{
		LLM:     llm,
		Storage: storage,
	})

	// Default intervals should be 20 and 60
	_, should20 := summarizer.ShouldSummarize(20)
	if !should20 {
		t.Error("Expected summary at message 20 with default intervals")
	}

	_, should60 := summarizer.ShouldSummarize(60)
	if !should60 {
		t.Error("Expected summary at message 60 with default intervals")
	}
}

func TestSummarizer_StoreAndGetBothSummaries(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	llm := testutil.NewFakeLLM()
	summarizer := NewSummarizer(SummarizerConfig{
		LLM:     llm,
		Storage: storage,
	})

	now := time.Now().Truncate(time.Second)

	// Store a short summary
	shortSummary := &Summary{
		Type:      SummaryTypeShort,
		Content:   "Short summary of conversation",
		MessageID: "msg-20",
		CreatedAt: now,
	}
	if err := summarizer.StoreSummary(ctx, "sess-1", shortSummary); err != nil {
		t.Fatalf("StoreSummary(short) error = %v", err)
	}

	// Store a long summary
	longSummary := &Summary{
		Type:      SummaryTypeLong,
		Content:   "Long summary of conversation",
		MessageID: "msg-60",
		CreatedAt: now,
	}
	if err := summarizer.StoreSummary(ctx, "sess-1", longSummary); err != nil {
		t.Fatalf("StoreSummary(long) error = %v", err)
	}

	// Retrieve both
	gotShort, gotLong, err := summarizer.GetBothSummaries(ctx, "sess-1")
	if err != nil {
		t.Fatalf("GetBothSummaries() error = %v", err)
	}

	if gotShort == nil {
		t.Fatal("Expected non-nil short summary")
	}
	if gotShort.Content != "Short summary of conversation" {
		t.Errorf("Short content = %q, want 'Short summary of conversation'", gotShort.Content)
	}
	if gotShort.Type != SummaryTypeShort {
		t.Errorf("Short type = %q, want %q", gotShort.Type, SummaryTypeShort)
	}

	if gotLong == nil {
		t.Fatal("Expected non-nil long summary")
	}
	if gotLong.Content != "Long summary of conversation" {
		t.Errorf("Long content = %q, want 'Long summary of conversation'", gotLong.Content)
	}
	if gotLong.Type != SummaryTypeLong {
		t.Errorf("Long type = %q, want %q", gotLong.Type, SummaryTypeLong)
	}
}

func TestSummarizer_StoreAndGetSummaries_ContentWithoutSummaryWord(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	llm := testutil.NewFakeLLM()
	summarizer := NewSummarizer(SummarizerConfig{
		LLM:     llm,
		Storage: storage,
	})

	now := time.Now().Truncate(time.Second)

	// Store a summary whose content does NOT contain the word "summary"
	// This verifies that GetBothSummaries finds it via summary_marker, not by content text.
	shortSummary := &Summary{
		Type:      SummaryTypeShort,
		Content:   "The user discussed programming languages and expressed interest in Go",
		MessageID: "msg-20",
		CreatedAt: now,
	}
	if err := summarizer.StoreSummary(ctx, "sess-1", shortSummary); err != nil {
		t.Fatalf("StoreSummary(short) error = %v", err)
	}

	gotShort, gotLong, err := summarizer.GetBothSummaries(ctx, "sess-1")
	if err != nil {
		t.Fatalf("GetBothSummaries() error = %v", err)
	}

	if gotShort == nil {
		t.Fatal("Expected non-nil short summary even when content lacks 'summary' word")
	}
	if gotShort.Content != "The user discussed programming languages and expressed interest in Go" {
		t.Errorf("Short content = %q, want original content", gotShort.Content)
	}
	if gotLong != nil {
		t.Error("Expected nil long summary when only short was stored")
	}
}

func TestSummarizer_GetBothSummaries_NoSummaries(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	llm := testutil.NewFakeLLM()
	summarizer := NewSummarizer(SummarizerConfig{
		LLM:     llm,
		Storage: storage,
	})

	gotShort, gotLong, err := summarizer.GetBothSummaries(ctx, "nonexistent-session")
	if err != nil {
		t.Fatalf("GetBothSummaries() error = %v", err)
	}
	if gotShort != nil {
		t.Error("Expected nil short summary for nonexistent session")
	}
	if gotLong != nil {
		t.Error("Expected nil long summary for nonexistent session")
	}
}

func TestSummarizer_Summarize_LLMError(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	llm := testutil.NewFakeLLM()
	llm.SetError(fmt.Errorf("LLM error"))
	summarizer := NewSummarizer(SummarizerConfig{
		LLM:     llm,
		Storage: storage,
	})

	msgs := []TimestampedMessage{
		{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "test"}}}, At: time.Now()},
	}

	_, err := summarizer.Summarize(ctx, msgs)
	if err == nil {
		t.Error("Expected error when LLM returns an error")
	}
}

func TestSummarizer_Summarize_NilLLM(t *testing.T) {
	ctx := context.Background()
	storage := adapter.InMemory()
	defer storage.Close()

	summarizer := NewSummarizer(SummarizerConfig{LLM: nil, Storage: storage})

	msgs := []TimestampedMessage{
		{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "test"}}}, At: time.Now()},
	}

	_, err := summarizer.Summarize(ctx, msgs)
	if err == nil {
		t.Fatal("Expected error for Summarize with nil LLM")
	}
	if !strings.Contains(err.Error(), "LLM is required") {
		t.Errorf("Error = %q, want error mentioning 'LLM is required'", err.Error())
	}
}

// Phase 3 Enhanced Summarizer Tests

func TestSummarizer_ShouldSummarizeEnhanced_MaxEvents(t *testing.T) {
	cfg := SummarizerConfig{
		ShortInterval: 20,
		LongInterval:  60,
		MaxEvents:     100,
	}
	s := NewSummarizer(cfg)

	tests := []struct {
		name            string
		messageCount    int
		estimatedTokens int
		wantType        SummaryType
		wantBool        bool
		wantReason      string
	}{
		{
			name:            "below MaxEvents threshold",
			messageCount:    50,
			estimatedTokens: 1000,
			wantType:        "",
			wantBool:        false,
		},
		{
			name:            "at MaxEvents threshold triggers interval",
			messageCount:    100, // At threshold but not above, but 100 % 20 == 0 triggers short
			estimatedTokens: 1000,
			wantType:        SummaryTypeShort,
			wantBool:        true,
			wantReason:      "short interval",
		},
		{
			name:            "exceeds MaxEvents threshold",
			messageCount:    101,
			estimatedTokens: 1000,
			wantType:        SummaryTypeLong,
			wantBool:        true,
			wantReason:      "exceeds MaxEvents",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotType, gotReason, gotBool := s.ShouldSummarizeEnhanced(tt.messageCount, tt.estimatedTokens)
			if gotBool != tt.wantBool {
				t.Errorf("ShouldSummarizeEnhanced() bool = %v, want %v", gotBool, tt.wantBool)
			}
			if gotType != tt.wantType {
				t.Errorf("ShouldSummarizeEnhanced() type = %q, want %q", gotType, tt.wantType)
			}
			if tt.wantReason != "" && !strings.Contains(gotReason, tt.wantReason) {
				t.Errorf("ShouldSummarizeEnhanced() reason = %q, want to contain %q", gotReason, tt.wantReason)
			}
		})
	}
}

func TestSummarizer_ShouldSummarizeEnhanced_MaxTokens(t *testing.T) {
	cfg := SummarizerConfig{
		ShortInterval: 20,
		LongInterval:  60,
		MaxTokens:     4000,
	}
	s := NewSummarizer(cfg)

	tests := []struct {
		name            string
		messageCount    int
		estimatedTokens int
		wantType        SummaryType
		wantBool        bool
		wantReason      string
	}{
		{
			name:            "below MaxTokens threshold",
			messageCount:    50,
			estimatedTokens: 2000,
			wantType:        "",
			wantBool:        false,
		},
		{
			name:            "exceeds MaxTokens threshold",
			messageCount:    50,
			estimatedTokens: 4001,
			wantType:        SummaryTypeLong,
			wantBool:        true,
			wantReason:      "exceeds MaxTokens",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotType, gotReason, gotBool := s.ShouldSummarizeEnhanced(tt.messageCount, tt.estimatedTokens)
			if gotBool != tt.wantBool {
				t.Errorf("ShouldSummarizeEnhanced() bool = %v, want %v", gotBool, tt.wantBool)
			}
			if gotType != tt.wantType {
				t.Errorf("ShouldSummarizeEnhanced() type = %q, want %q", gotType, tt.wantType)
			}
			if tt.wantReason != "" && !strings.Contains(gotReason, tt.wantReason) {
				t.Errorf("ShouldSummarizeEnhanced() reason = %q, want to contain %q", gotReason, tt.wantReason)
			}
		})
	}
}

func TestSummarizer_ShouldSummarizeEnhanced_IntervalFallback(t *testing.T) {
	cfg := SummarizerConfig{
		ShortInterval: 20,
		LongInterval:  60,
	}
	s := NewSummarizer(cfg)

	tests := []struct {
		name            string
		messageCount    int
		estimatedTokens int
		wantType        SummaryType
		wantBool        bool
		wantReason      string
	}{
		{
			name:            "short interval triggered",
			messageCount:    20,
			estimatedTokens: 100,
			wantType:        SummaryTypeShort,
			wantBool:        true,
			wantReason:      "short interval",
		},
		{
			name:            "long interval triggered",
			messageCount:    60,
			estimatedTokens: 100,
			wantType:        SummaryTypeLong,
			wantBool:        true,
			wantReason:      "long interval",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotType, gotReason, gotBool := s.ShouldSummarizeEnhanced(tt.messageCount, tt.estimatedTokens)
			if gotBool != tt.wantBool {
				t.Errorf("ShouldSummarizeEnhanced() bool = %v, want %v", gotBool, tt.wantBool)
			}
			if gotType != tt.wantType {
				t.Errorf("ShouldSummarizeEnhanced() type = %q, want %q", gotType, tt.wantType)
			}
			if tt.wantReason != "" && !strings.Contains(gotReason, tt.wantReason) {
				t.Errorf("ShouldSummarizeEnhanced() reason = %q, want to contain %q", gotReason, tt.wantReason)
			}
		})
	}
}

func TestEstimateTokens(t *testing.T) {
	tests := []struct {
		name     string
		messages []TimestampedMessage
		wantMin  int // Allow some tolerance
		wantMax  int
	}{
		{
			name:     "empty messages",
			messages: []TimestampedMessage{},
			wantMin:  0,
			wantMax:  0,
		},
		{
			name: "single short message",
			messages: []TimestampedMessage{
				{Content: &genai.Content{Parts: []*genai.Part{{Text: "hello"}}}},
			},
			wantMin: 1,
			wantMax: 2,
		},
		{
			name: "multiple messages",
			messages: []TimestampedMessage{
				{Content: &genai.Content{Parts: []*genai.Part{{Text: "hello world"}}}},
				{Content: &genai.Content{Parts: []*genai.Part{{Text: "test message"}}}},
			},
			wantMin: 5,
			wantMax: 6,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EstimateTokens(tt.messages)
			if got < tt.wantMin || got > tt.wantMax {
				t.Errorf("EstimateTokens() = %d, want between %d and %d", got, tt.wantMin, tt.wantMax)
			}
		})
	}
}

func TestSummarizer_GetMessagesForSummarization(t *testing.T) {
	createMessages := func(n int) []TimestampedMessage {
		msgs := make([]TimestampedMessage, n)
		for i := 0; i < n; i++ {
			msgs[i] = TimestampedMessage{
				Content: &genai.Content{Parts: []*genai.Part{{Text: fmt.Sprintf("message %d", i)}}},
			}
		}
		return msgs
	}

	tests := []struct {
		name       string
		keepRecent int
		msgCount   int
		wantCount  int
	}{
		{
			name:       "KeepRecent 0 returns all",
			keepRecent: 0,
			msgCount:   10,
			wantCount:  10,
		},
		{
			name:       "KeepRecent 2 excludes last 2",
			keepRecent: 2,
			msgCount:   10,
			wantCount:  8,
		},
		{
			name:       "KeepRecent larger than messages returns all",
			keepRecent: 20,
			msgCount:   10,
			wantCount:  10,
		},
		{
			name:       "KeepRecent equals messages returns all",
			keepRecent: 10,
			msgCount:   10,
			wantCount:  10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := SummarizerConfig{
				KeepRecent: tt.keepRecent,
			}
			s := NewSummarizer(cfg)
			msgs := createMessages(tt.msgCount)
			got := s.GetMessagesForSummarization(msgs)
			if len(got) != tt.wantCount {
				t.Errorf("GetMessagesForSummarization() returned %d messages, want %d", len(got), tt.wantCount)
			}
		})
	}
}
