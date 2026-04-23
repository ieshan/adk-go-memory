package memory

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"iter"
	"time"

	"github.com/ieshan/adk-go-memory/adapter"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// TimestampedMessage pairs a genai content with its wall-clock timestamp.
type TimestampedMessage struct {
	Content *genai.Content
	At      time.Time
}

// DeriverConfig configures the Deriver.
type DeriverConfig struct {
	// LLM is used to extract observations from conversation messages.
	LLM model.LLM
	// Storage is the database adapter for saving observations.
	Storage adapter.Storage
	// EmbeddingFunc generates embeddings for deduplication search.
	// When provided, vector-based deduplication is enabled alongside FTS.
	// When nil, deduplication relies on FTS text matching only.
	EmbeddingFunc func(ctx context.Context, text string) ([]float32, error)
}

// Deriver extracts factual observations from conversations using an LLM.
// It analyzes messages and produces Observation values at various
// confidence levels (explicit, deductive, inductive, contradiction).
// Near-duplicate observations are detected via hybrid search and deduplicated
// by incrementing the existing observation's TimesDerived counter.
type Deriver struct {
	llm           model.LLM
	storage       adapter.Storage
	embeddingFunc func(ctx context.Context, text string) ([]float32, error)
}

// deriverResponse is the expected JSON structure from the LLM.
type deriverResponse struct {
	Observations []deriverObs `json:"observations"`
}

type deriverObs struct {
	Content    string   `json:"content"`
	Level      string   `json:"level"`
	Confidence float64  `json:"confidence"` // 0 = use level-based default
	Tags       []string `json:"tags"`
}

const deriverSystemPrompt = `You are a memory extraction agent. Analyze the conversation and extract factual observations about the user.

Return a JSON object with this exact structure:
{"observations": [{"content": "observation text", "level": "explicit|deductive|inductive|contradiction", "confidence": 0.95, "tags": ["tag1", "tag2"]}]}

Observation levels:
- explicit: directly stated facts
- deductive: logical inferences from stated facts
- inductive: patterns from 3+ observations
- contradiction: conflicting statements

Use the absolute date from message timestamps when recording facts about time-sensitive information.
Return an empty observations array if no new facts are found.`

const (
	// dedupVectorThreshold is the minimum cosine similarity for a vector result
	// to be considered a near-duplicate.
	dedupVectorThreshold = 0.92
	// dedupFTSThreshold is the minimum BM25-derived score for an FTS-only result
	// to be considered a near-duplicate.
	dedupFTSThreshold = 0.85
	// dedupRRFThreshold is the minimum RRF score for a hybrid result to be
	// considered a near-duplicate. RRF scores are bounded by 2/(k+1) ≈ 0.0328
	// for k=60 (item ranks #1 in both lists). This threshold requires the item
	// to rank in the top 3 of both constituent lists, ensuring strong agreement
	// between vector and FTS before considering a duplicate.
	dedupRRFThreshold = 2.0 / (60.0 + 3.0)
)

// NewDeriver creates a new Deriver with the given configuration.
func NewDeriver(cfg DeriverConfig) *Deriver {
	return &Deriver{llm: cfg.LLM, storage: cfg.Storage, embeddingFunc: cfg.EmbeddingFunc}
}

// Derive extracts observations from timestamped messages and stores them.
// userID and appName are stored on each observation for filtering.
func (d *Deriver) Derive(ctx context.Context, messages []TimestampedMessage, sessionID, userID, appName string) error {
	if len(messages) == 0 {
		return nil
	}
	if d.llm == nil {
		return fmt.Errorf("deriver: LLM is required for observation extraction")
	}

	// Build LLM request with system prompt
	contents := []*genai.Content{
		{
			Role:  "system",
			Parts: []*genai.Part{{Text: deriverSystemPrompt}},
		},
	}
	for _, tm := range messages {
		formatted := formatTimestampedMessage(tm)
		contents = append(contents, &genai.Content{
			Role:  tm.Content.Role,
			Parts: []*genai.Part{{Text: formatted}},
		})
	}

	req := &model.LLMRequest{Contents: contents}

	// Call LLM
	responseText := ""
	for resp, err := range d.llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return fmt.Errorf("deriver: LLM call: %w", err)
		}
		if resp != nil && resp.Content != nil {
			for _, p := range resp.Content.Parts {
				if p.Text != "" {
					responseText += p.Text
				}
			}
		}
	}

	if responseText == "" {
		return nil
	}

	// Parse response
	var result deriverResponse
	if err := json.Unmarshal([]byte(responseText), &result); err != nil {
		return fmt.Errorf("deriver: parse response: %w", err)
	}

	// Store observations
	for _, obs := range result.Observations {
		// Validate level — default to inductive if unrecognized
		level := obs.Level
		if !validLevel(level) {
			level = string(adapter.LevelInductive)
		}

		id, err := randomID("obs")
		if err != nil {
			return fmt.Errorf("deriver: generate ID: %w", err)
		}

		// Deduplication check
		dupeID, err := d.findNearDuplicate(ctx, obs.Content, sessionID)
		if err != nil {
			return fmt.Errorf("deriver: dedup search: %w", err)
		}
		if dupeID != "" {
			if err := d.storage.IncrementTimesDerived(ctx, dupeID); err != nil {
				return fmt.Errorf("deriver: increment times_derived: %w", err)
			}
			continue
		}

		// Store new observation
		if err := d.storage.Store(ctx, &adapter.Observation{
			ID:           id,
			Content:      obs.Content,
			Level:        adapter.ObservationLevel(level),
			SessionID:    sessionID,
			UserID:       userID,
			AppName:      appName,
			TimesDerived: 1,
			CreatedAt:    time.Now(),
			Tags:         obs.Tags,
		}); err != nil {
			return fmt.Errorf("deriver: store: %w", err)
		}
	}

	return nil
}

// findNearDuplicate searches for an existing observation whose content is
// semantically close to content. Returns the ID of the near-duplicate if
// found above the configured threshold, or "" if none found.
// Uses hybrid search and applies score thresholds for robust deduplication.
// When an EmbeddingFunc is configured, the embedding is provided to enable
// vector-based deduplication alongside FTS.
func (d *Deriver) findNearDuplicate(ctx context.Context, content, sessionID string) (string, error) {
	opts := &adapter.SearchOptions{
		Query:      content,
		MaxResults: 3,
		Mode:       adapter.SearchModeHybrid,
		SessionID:  sessionID,
	}

	// If an embedding function is available, generate an embedding for the content
	// so that vector-based deduplication actually works.
	if d.embeddingFunc != nil {
		if embedding, err := d.embeddingFunc(ctx, content); err == nil && len(embedding) > 0 {
			opts.Embedding = embedding
		}
	}

	results, err := d.storage.Search(ctx, opts)
	if err != nil {
		return "", err
	}
	if len(results) == 0 {
		return "", nil
	}

	// Apply threshold-based deduplication
	// Vector fallback results are recent observations without semantic matching
	// and should not be treated as near-duplicates.
	for _, r := range results {
		switch r.Source {
		case "rrf":
			if r.Score >= dedupRRFThreshold {
				return r.Observation.ID, nil
			}
		case "fts":
			if r.Score >= dedupFTSThreshold {
				return r.Observation.ID, nil
			}
		case "vector":
			if r.Score >= dedupVectorThreshold {
				return r.Observation.ID, nil
			}
		case "vector_fallback":
			// Recent observations without semantic matching —
			// cannot confirm near-duplication, so skip.
		}
	}

	return "", nil
}

// validLevel checks if the given level is a recognized ObservationLevel.
func validLevel(level string) bool {
	switch adapter.ObservationLevel(level) {
	case adapter.LevelExplicit, adapter.LevelDeductive, adapter.LevelInductive, adapter.LevelContradiction:
		return true
	default:
		return false
	}
}

// formatTimestampedMessage formats a message as "[timestamp] role: text"
func formatTimestampedMessage(tm TimestampedMessage) string {
	ts := tm.At.UTC().Format(time.RFC3339)
	var text string
	for _, p := range tm.Content.Parts {
		if p.Text != "" {
			text += p.Text
		}
	}
	role := tm.Content.Role
	if role == "" {
		role = "user"
	}
	return "[" + ts + "] " + role + ": " + text
}

// randomID generates a cryptographically random ID with the given prefix.
func randomID(prefix string) (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return prefix + "-" + hex.EncodeToString(b[:]), nil
}

// fakeLLM is a test helper that implements model.LLM for testing.
type fakeLLM struct {
	responses []model.LLMResponse
	calls     []*model.LLMRequest
}

func (f *fakeLLM) Name() string { return "fake-llm" }

func (f *fakeLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	f.calls = append(f.calls, req)
	return func(yield func(*model.LLMResponse, error) bool) {
		if len(f.responses) == 0 {
			yield(nil, fmt.Errorf("fakeLLM: no responses configured"))
			return
		}
		resp := &f.responses[0]
		f.responses = f.responses[1:]
		yield(resp, nil)
	}
}
