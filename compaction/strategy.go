// Package compaction provides session event compaction strategies for adk-go-memory.
package compaction

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// Strategy defines the interface for session event compaction algorithms.
// Implementations transform a slice of events into a compacted representation.
type Strategy interface {
	// Compact transforms the input events into a compacted form.
	// The returned events replace the original events in the session context.
	//
	// For truncation strategies, this may return fewer events.
	// For summarization strategies, this may return a single summary event.
	//
	// The context is passed for cancellation and timeout handling.
	Compact(ctx context.Context, events []*session.Event) ([]*session.Event, error)
}

// TruncationStrategy keeps only the most recent N events.
// This is a simple, fast strategy that doesn't require an LLM.
type TruncationStrategy struct {
	// RetainCount is the number of most recent events to keep.
	// Must be > 0.
	RetainCount int
}

var _ Strategy = (*TruncationStrategy)(nil)

// Compact implements the Strategy interface.
// It keeps only the last RetainCount events from the input.
func (t *TruncationStrategy) Compact(ctx context.Context, events []*session.Event) ([]*session.Event, error) {
	if t.RetainCount <= 0 {
		return nil, fmt.Errorf("compaction: truncate: RetainCount must be > 0, got %d", t.RetainCount)
	}

	if len(events) <= t.RetainCount {
		// Nothing to truncate
		return events, nil
	}

	// Keep only the last RetainCount events
	startIdx := len(events) - t.RetainCount
	return events[startIdx:], nil
}

// SummarizationStrategy uses an LLM to create a summary of events.
// The summary is stored as a single system-authored event.
type SummarizationStrategy struct {
	// LLM is the language model used for summarization.
	// Required.
	LLM model.LLM

	// Instruction is the prompt template for summarization.
	// If empty, a default prompt is used.
	Instruction string

	// MaxSummaryTokens is the maximum tokens for the generated summary.
	// 0 means use LLM default.
	MaxSummaryTokens int
}

var _ Strategy = (*SummarizationStrategy)(nil)
var _ Strategy = (*CompositeStrategy)(nil)

// Compact implements the Strategy interface.
// It generates a summary of all input events and returns it as a single event.
func (s *SummarizationStrategy) Compact(ctx context.Context, events []*session.Event) ([]*session.Event, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("compaction: compact: context cancelled: %w", err)
	}

	if s.LLM == nil {
		return nil, fmt.Errorf("compaction: summarize: LLM is required")
	}

	if len(events) == 0 {
		return events, nil
	}

	// Build prompt from events
	prompt := s.buildPrompt(events)

	// Generate summary using LLM
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText(prompt, genai.RoleUser)},
	}

	var response *model.LLMResponse
	for resp, err := range s.LLM.GenerateContent(ctx, req, false) {
		if err != nil {
			return nil, fmt.Errorf("compaction: summarize: LLM generation failed: %w", err)
		}
		response = resp
		break // Non-streaming, take first response
	}

	// Extract summary text from response
	summaryText := ExtractSummaryText(response)
	if summaryText == "" {
		return nil, fmt.Errorf("compaction: summarize: LLM returned empty summary")
	}

	// Create a summary event
	summaryEvent := &session.Event{
		Author:    "system",
		Timestamp: time.Now(),
	}
	summaryEvent.Content = genai.NewContentFromText(summaryText, genai.RoleModel)

	return []*session.Event{summaryEvent}, nil
}

// defaultSummaryPrompt is used when no custom instruction is provided.
const defaultSummaryPrompt = `You are a helpful assistant that summarizes conversation history.

Please provide a concise summary of the following conversation, capturing:
- Key facts and information shared
- User preferences and opinions
- Important context and requests
- Core topics discussed

Keep the summary factual and neutral. The summary should help provide context for future messages.

Conversation:
%s

Summary:`

// buildPrompt creates the summarization prompt from events.
func (s *SummarizationStrategy) buildPrompt(events []*session.Event) string {
	var conversation string
	for i, event := range events {
		author := event.Author
		if author == "" {
			author = "unknown"
		}

		content := ""
		if event.Content != nil {
			for _, part := range event.Content.Parts {
				if part != nil && part.Text != "" {
					content += part.Text
				}
			}
		}

		conversation += fmt.Sprintf("\n[%d] %s: %s\n", i+1, author, content)
	}

	instruction := s.Instruction
	if instruction == "" {
		instruction = defaultSummaryPrompt
	}

	return fmt.Sprintf(instruction, conversation)
}

// CompositeStrategy applies multiple strategies in sequence.
// Each strategy's output becomes the next strategy's input.
type CompositeStrategy struct {
	Strategies []Strategy
}

var _ Strategy = (*CompositeStrategy)(nil)

// Compact implements the Strategy interface.
// It applies each strategy in sequence.
func (c *CompositeStrategy) Compact(ctx context.Context, events []*session.Event) ([]*session.Event, error) {
	var result = events
	var err error

	for i, strategy := range c.Strategies {
		result, err = strategy.Compact(ctx, result)
		if err != nil {
			return nil, fmt.Errorf("compaction: composite strategy [%d]: %w", i, err)
		}
	}

	return result, nil
}

// IdentityStrategy is a no-op strategy that returns events unchanged.
// Useful for testing or when compaction is temporarily disabled.
type IdentityStrategy struct{}

var _ Strategy = (*IdentityStrategy)(nil)

// Compact implements the Strategy interface.
func (i *IdentityStrategy) Compact(ctx context.Context, events []*session.Event) ([]*session.Event, error) {
	return events, nil
}
