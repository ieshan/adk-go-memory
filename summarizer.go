package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/ieshan/adk-go-memory/adapter"
	"github.com/ieshan/idx"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// SummaryType represents the type of summary.
type SummaryType string

const (
	SummaryTypeShort SummaryType = "short"
	SummaryTypeLong  SummaryType = "long"
)

// SummarizerConfig configures the summarizer.
type SummarizerConfig struct {
	LLM           model.LLM
	Storage       adapter.Storage
	ShortInterval int // Messages per short summary (default: 20)
	LongInterval  int // Messages per long summary (default: 60)

	// Threshold-based triggers (Phase 3 enhancement)
	MaxEvents  int // Trigger when events exceed this (0 = disabled)
	MaxTokens  int // Trigger when estimated tokens exceed this (0 = disabled)
	KeepRecent int // Always preserve this many recent events (default: 10)
}

// Summarizer creates session summaries at defined intervals.
type Summarizer struct {
	llm     model.LLM
	storage adapter.Storage
	config  SummarizerConfig
}

// Summary represents a session summary.
type Summary struct {
	Type      SummaryType
	Content   string
	MessageID string
	CreatedAt time.Time
}

const summarizerSystemPrompt = `You are a summarization agent. Create a concise summary of the conversation.

Focus on:
- Key facts and decisions made
- User preferences revealed
- Action items or next steps

Keep the summary brief but comprehensive.`

// ApproximateTokensPerChar is the rough token-to-character ratio (4 chars ≈ 1 token).
const ApproximateTokensPerChar = 0.25

// NewSummarizer creates a new summarizer.
func NewSummarizer(cfg SummarizerConfig) *Summarizer {
	if cfg.ShortInterval == 0 {
		cfg.ShortInterval = 20
	}
	if cfg.LongInterval == 0 {
		cfg.LongInterval = 60
	}
	if cfg.KeepRecent < 0 {
		cfg.KeepRecent = 0
	}
	return &Summarizer{
		llm:     cfg.LLM,
		storage: cfg.Storage,
		config:  cfg,
	}
}

// ShouldSummarize checks if a summary should be created at this message count.
func (s *Summarizer) ShouldSummarize(messageCount int) (SummaryType, bool) {
	if messageCount > 0 && messageCount%s.config.LongInterval == 0 {
		return SummaryTypeLong, true
	}
	if messageCount > 0 && messageCount%s.config.ShortInterval == 0 {
		return SummaryTypeShort, true
	}
	return "", false
}

// ShouldSummarizeEnhanced checks both interval and threshold triggers.
// Returns the summary type and true if a summary should be created.
// This is the Phase 3 enhanced version supporting threshold-based triggers.
func (s *Summarizer) ShouldSummarizeEnhanced(messageCount, estimatedTokens int) (SummaryType, string, bool) {
	// Check threshold triggers first (they take precedence)
	if s.config.MaxEvents > 0 && messageCount > s.config.MaxEvents {
		return SummaryTypeLong, fmt.Sprintf("event count %d exceeds MaxEvents %d", messageCount, s.config.MaxEvents), true
	}

	if s.config.MaxTokens > 0 && estimatedTokens > s.config.MaxTokens {
		return SummaryTypeLong, fmt.Sprintf("estimated tokens %d exceeds MaxTokens %d", estimatedTokens, s.config.MaxTokens), true
	}

	// Fall back to interval-based triggers
	if messageCount > 0 && messageCount%s.config.LongInterval == 0 {
		return SummaryTypeLong, fmt.Sprintf("reached long interval at %d messages", messageCount), true
	}
	if messageCount > 0 && messageCount%s.config.ShortInterval == 0 {
		return SummaryTypeShort, fmt.Sprintf("reached short interval at %d messages", messageCount), true
	}

	return "", "", false
}

// EstimateTokens approximates the token count for a slice of timestamped messages.
// Uses a rough heuristic: 4 characters ≈ 1 token.
func EstimateTokens(messages []TimestampedMessage) int {
	var charCount int
	for _, msg := range messages {
		if msg.Content == nil {
			continue
		}
		for _, part := range msg.Content.Parts {
			charCount += len(part.Text)
		}
	}
	return int(float64(charCount) * ApproximateTokensPerChar)
}

// GetMessagesForSummarization returns messages to summarize, respecting KeepRecent.
// It returns messages excluding the most recent KeepRecent messages.
func (s *Summarizer) GetMessagesForSummarization(messages []TimestampedMessage) []TimestampedMessage {
	if s.config.KeepRecent <= 0 || len(messages) <= s.config.KeepRecent {
		return messages
	}
	return messages[:len(messages)-s.config.KeepRecent]
}

// Summarize creates a summary of the given messages.
func (s *Summarizer) Summarize(ctx context.Context, messages []TimestampedMessage) (*Summary, error) {
	if len(messages) == 0 {
		return nil, fmt.Errorf("summarizer: no messages to summarize")
	}
	if s.llm == nil {
		return nil, fmt.Errorf("summarizer: LLM is required for summarization")
	}

	// Build transcript
	var transcript strings.Builder
	for _, msg := range messages {
		formatted := formatTimestampedMessage(msg)
		transcript.WriteString(formatted)
		transcript.WriteString("\n")
	}

	contents := []*genai.Content{
		{
			Role:  "system",
			Parts: []*genai.Part{{Text: summarizerSystemPrompt}},
		},
		{
			Role:  "user",
			Parts: []*genai.Part{{Text: "Summarize this conversation:\n\n" + transcript.String()}},
		},
	}

	req := &model.LLMRequest{Contents: contents}

	var summaryText string
	for resp, err := range s.llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return nil, fmt.Errorf("summarizer: LLM call: %w", err)
		}
		if resp != nil && resp.Content != nil {
			for _, p := range resp.Content.Parts {
				if p.Text != "" {
					summaryText += p.Text
				}
			}
		}
	}

	return &Summary{
		Content:   summaryText,
		CreatedAt: time.Now(),
	}, nil
}

// StoreSummary persists a summary to the storage (as a special observation).
func (s *Summarizer) StoreSummary(ctx context.Context, sessionID string, summary *Summary) error {
	id := idx.NewID()

	content, _ := json.Marshal(map[string]interface{}{
		"summary_marker": true,
		"type":           summary.Type,
		"content":        summary.Content,
		"message_id":     summary.MessageID,
	})

	obs := &adapter.Observation{
		ID:           id,
		Content:      string(content),
		Level:        adapter.LevelExplicit,
		SessionID:    sessionID,
		Tags:         []string{"summary", string(summary.Type)},
		TimesDerived: 1,
		CreatedAt:    summary.CreatedAt,
	}

	return s.storage.Store(ctx, obs)
}

// summaryData is the JSON structure stored in observations for summaries.
type summaryData struct {
	Type      SummaryType `json:"type"`
	Content   string      `json:"content"`
	MessageID string      `json:"message_id"`
}

// GetBothSummaries retrieves short and long summaries for a session.
func (s *Summarizer) GetBothSummaries(ctx context.Context, sessionID string) (*Summary, *Summary, error) {
	// Search for summary observations using the summary_marker field
	// that is always present in stored summary JSON content.
	results, err := s.storage.Search(ctx, &adapter.SearchOptions{
		Query:      "summary_marker",
		MaxResults: 20,
		Mode:       adapter.SearchModeFTS,
		SessionID:  sessionID,
	})
	if err != nil {
		return nil, nil, err
	}

	var shortSum, longSum *Summary
	for _, r := range results {
		var data summaryData
		if err := json.Unmarshal([]byte(r.Observation.Content), &data); err != nil {
			continue
		}
		sum := &Summary{
			Type:      data.Type,
			Content:   data.Content,
			MessageID: data.MessageID,
			CreatedAt: r.Observation.CreatedAt,
		}
		switch data.Type {
		case SummaryTypeShort:
			shortSum = sum
		case SummaryTypeLong:
			longSum = sum
		}
	}

	return shortSum, longSum, nil
}
