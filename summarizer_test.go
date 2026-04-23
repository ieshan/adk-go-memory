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
	storage, _ := adapter.InMemory()
	defer storage.Close()

	llm := &fakeLLM{
		responses: []model.LLMResponse{{
			Content: &genai.Content{
				Parts: []*genai.Part{{
					Text: "Summary of conversation about Go programming.",
				}},
			},
		}},
	}

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
	storage, _ := adapter.InMemory()
	defer storage.Close()

	llm := &fakeLLM{}
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
	storage, _ := adapter.InMemory()
	defer storage.Close()

	llm := &fakeLLM{}
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
	storage, _ := adapter.InMemory()
	defer storage.Close()

	llm := &fakeLLM{}
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
	storage, _ := adapter.InMemory()
	defer storage.Close()

	llm := &fakeLLM{}
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
	storage, _ := adapter.InMemory()
	defer storage.Close()

	llm := &fakeLLM{}
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
	storage, _ := adapter.InMemory()
	defer storage.Close()

	llm := &fakeLLM{} // No responses configured -> returns error
	summarizer := NewSummarizer(SummarizerConfig{
		LLM:     llm,
		Storage: storage,
	})

	msgs := []TimestampedMessage{
		{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "test"}}}, At: time.Now()},
	}

	_, err := summarizer.Summarize(ctx, msgs)
	if err == nil {
		t.Error("Expected error when LLM has no responses")
	}
}

func TestSummarizer_Summarize_NilLLM(t *testing.T) {
	ctx := context.Background()
	storage, _ := adapter.InMemory()
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
