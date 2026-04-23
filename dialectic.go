package memory

import (
	"context"
	"fmt"
	"strings"

	"github.com/ieshan/adk-go-memory/adapter"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// DialecticConfig configures the Dialectic.
type DialecticConfig struct {
	LLM        model.LLM
	Storage    adapter.Storage
	MaxResults int
}

// QueryOptions configures a query operation.
type QueryOptions struct {
	SessionID string
	UserID    string
	AppName   string
}

// Dialectic provides LLM-powered memory query capabilities.
// It searches memory context and synthesizes answers via LLM reasoning.
type Dialectic struct {
	llm        model.LLM
	storage    adapter.Storage
	maxResults int
}

// NewDialectic creates a new Dialectic with the given configuration.
func NewDialectic(cfg DialecticConfig) *Dialectic {
	maxResults := cfg.MaxResults
	if maxResults == 0 {
		maxResults = 10
	}
	return &Dialectic{
		llm:        cfg.LLM,
		storage:    cfg.Storage,
		maxResults: maxResults,
	}
}

// Query performs a memory-based query and returns a synthesized answer.
func (d *Dialectic) Query(ctx context.Context, query string, opts QueryOptions) (string, error) {
	if d.storage == nil {
		return "", nil
	}
	if d.llm == nil {
		return "", fmt.Errorf("dialectic: LLM is required for query")
	}

	// Search for relevant observations
	results, err := d.storage.Search(ctx, &adapter.SearchOptions{
		Query:      query,
		MaxResults: d.maxResults,
		Mode:       adapter.SearchModeHybrid,
		SessionID:  opts.SessionID,
		UserID:     opts.UserID,
		AppName:    opts.AppName,
	})
	if err != nil {
		return "", fmt.Errorf("dialectic: search: %w", err)
	}

	// Build context from observations
	var contextBuilder strings.Builder
	contextBuilder.WriteString("Memory context:\n")
	if len(results) == 0 {
		contextBuilder.WriteString("(No relevant memories found)\n")
	} else {
		for _, r := range results {
			contextBuilder.WriteString(fmt.Sprintf("- %s (level: %s)\n", r.Observation.Content, r.Observation.Level))
		}
	}

	// Call LLM to synthesize answer
	contents := []*genai.Content{
		{
			Role:  "system",
			Parts: []*genai.Part{{Text: "You are a helpful assistant. Answer the user's question using the provided memory context."}},
		},
		{
			Role:  "user",
			Parts: []*genai.Part{{Text: contextBuilder.String() + "\nQuestion: " + query}},
		},
	}

	req := &model.LLMRequest{Contents: contents}

	var answer string
	for resp, err := range d.llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return "", fmt.Errorf("dialectic: LLM call: %w", err)
		}
		if resp != nil && resp.Content != nil {
			for _, p := range resp.Content.Parts {
				if p.Text != "" {
					answer += p.Text
				}
			}
		}
	}

	return answer, nil
}
