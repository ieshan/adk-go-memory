package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/ieshan/adk-go-memory/adapter"
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
}

// Summarizer creates session summaries at defined intervals.
type Summarizer struct {
	llm     model.LLM
	storage adapter.Storage
	config  SummarizerConfig
}

// Summary represents a session summary.
type Summary struct {
	Type       SummaryType
	Content    string
	MessageID  string
	TokenCount int
	CreatedAt  time.Time
}

const summarizerSystemPrompt = `You are a summarization agent. Create a concise summary of the conversation.

Focus on:
- Key facts and decisions made
- User preferences revealed
- Action items or next steps

Keep the summary brief but comprehensive.`

// NewSummarizer creates a new summarizer.
func NewSummarizer(cfg SummarizerConfig) *Summarizer {
	if cfg.ShortInterval == 0 {
		cfg.ShortInterval = 20
	}
	if cfg.LongInterval == 0 {
		cfg.LongInterval = 60
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
	id, err := randomID("summary")
	if err != nil {
		return err
	}

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
