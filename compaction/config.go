package compaction

import (
	"fmt"
	"time"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// Config holds the configuration for session event compaction.
type Config struct {
	// Strategy is the compaction algorithm to use.
	// Required. Common choices:
	//   - TruncationStrategy: Fast, keeps last N events
	//   - SummarizationStrategy: LLM-based, creates summary
	//   - CompositeStrategy: Chain multiple strategies
	Strategy Strategy

	// MaxEvents triggers compaction when event count exceeds this threshold.
	// 0 = disabled (no event count triggering).
	MaxEvents int

	// MaxTokens triggers compaction when estimated token count exceeds this threshold.
	// 0 = disabled (no token-based triggering).
	// Token estimation is approximate (4 chars ≈ 1 token).
	MaxTokens int

	// Interval triggers compaction every N events.
	// 0 = disabled (no interval-based triggering).
	// For example, Interval: 50 compacts every 50 events.
	Interval int

	// KeepRecent specifies how many recent events to always preserve.
	// These events are never compacted and remain in the session.
	// Must be >= 0. Default: 10.
	KeepRecent int

	// EnableAfterAgent enables post-run compaction check.
	// When true, compaction is checked again after the agent finishes.
	// Useful for long-running agents that generate many events.
	EnableAfterAgent bool

	// stateKey is the session.State key for tracking compaction state.
	// If empty, uses DefaultStateKey.
	stateKey string
}

const (
	// DefaultStateKey is the default session.State key for compaction state.
	DefaultStateKey = "adk_memory_compaction_state"

	// DefaultKeepRecent is the default number of events to preserve.
	DefaultKeepRecent = 10

	// ApproximateTokensPerChar is the rough token-to-character ratio.
	ApproximateTokensPerChar = 0.25 // 4 chars ≈ 1 token
)

// state represents the compaction state stored in session.State.
type state struct {
	// LastCompactedIndex is the index of the last event included in a compaction summary.
	// Events after this index are considered "new" and not yet compacted.
	LastCompactedIndex int `json:"last_compacted_index"`

	// LastCompactionTime is when the last compaction occurred.
	LastCompactionTime time.Time `json:"last_compaction_time"`

	// TotalCompactions is the total number of compactions performed.
	TotalCompactions int `json:"total_compactions"`

	// LastEventCount is the event count at last compaction (for interval tracking).
	LastEventCount int `json:"last_event_count"`
}

// GetLastCompactedIndex returns the index of the last compacted event.
func (s *state) GetLastCompactedIndex() int {
	return s.LastCompactedIndex
}

// Validate checks the configuration for errors.
func (c *Config) Validate() error {
	if c.Strategy == nil {
		return fmt.Errorf("compaction: validate: Strategy is required")
	}

	if c.MaxEvents < 0 {
		return fmt.Errorf("compaction: validate: MaxEvents must be >= 0, got %d", c.MaxEvents)
	}

	if c.MaxTokens < 0 {
		return fmt.Errorf("compaction: validate: MaxTokens must be >= 0, got %d", c.MaxTokens)
	}

	if c.Interval < 0 {
		return fmt.Errorf("compaction: validate: Interval must be >= 0, got %d", c.Interval)
	}

	if c.KeepRecent < 0 {
		return fmt.Errorf("compaction: validate: KeepRecent must be >= 0, got %d", c.KeepRecent)
	}

	// At least one trigger must be enabled
	if c.MaxEvents == 0 && c.MaxTokens == 0 && c.Interval == 0 {
		return fmt.Errorf("compaction: validate: at least one trigger (MaxEvents, MaxTokens, Interval) must be enabled")
	}

	return nil
}

// StateKey returns the configured state key, or the default if not set.
func (c *Config) StateKey() string {
	if c.stateKey != "" {
		return c.stateKey
	}
	return DefaultStateKey
}

// ShouldCompact determines if compaction should occur based on the current session state.
// Returns true if any trigger condition is met.
func (c *Config) ShouldCompact(sess session.Session, compactionState *state) (bool, string) {
	eventCount := sess.Events().Len()

	// Check MaxEvents trigger
	if c.MaxEvents > 0 && eventCount > c.MaxEvents {
		return true, fmt.Sprintf("event count %d exceeds MaxEvents %d", eventCount, c.MaxEvents)
	}

	// Check Interval trigger
	if c.Interval > 0 {
		newEventsSinceLastCompaction := eventCount - compactionState.LastEventCount
		if newEventsSinceLastCompaction >= c.Interval {
			return true, fmt.Sprintf("interval reached: %d new events since last compaction", newEventsSinceLastCompaction)
		}
	}

	// Check MaxTokens trigger
	if c.MaxTokens > 0 {
		estimatedTokens := c.estimateTokens(sess, compactionState.LastCompactedIndex)
		if estimatedTokens > c.MaxTokens {
			return true, fmt.Sprintf("estimated tokens %d exceeds MaxTokens %d", estimatedTokens, c.MaxTokens)
		}
	}

	return false, ""
}

// estimateTokens approximates the token count for events after startIndex.
// Uses a rough heuristic: 4 characters ≈ 1 token.
func (c *Config) estimateTokens(sess session.Session, startIndex int) int {
	events := sess.Events()
	var charCount int

	for i := startIndex; i < events.Len(); i++ {
		event := events.At(i)
		if event == nil || event.Content == nil {
			continue
		}

		for _, part := range event.Content.Parts {
			if part != nil {
				charCount += len(part.Text)
			}
		}
	}

	return int(float64(charCount) * ApproximateTokensPerChar)
}

// GetCompactionState retrieves the compaction state from session.State.
// Returns a zero-valued state if not found.
func GetCompactionState(sess session.Session, key string) (*state, error) {
	if key == "" {
		key = DefaultStateKey
	}

	value, err := sess.State().Get(key)
	if err != nil {
		// State key not found - return zero state
		return &state{}, nil
	}

	// Try to convert to state struct
	s, ok := value.(*state)
	if !ok {
		return nil, fmt.Errorf("compaction: get state: invalid type %T", value)
	}

	return s, nil
}

// SaveCompactionState saves the compaction state to session.State.
func SaveCompactionState(sess session.Session, key string, st *state) error {
	if key == "" {
		key = DefaultStateKey
	}
	return sess.State().Set(key, st)
}

// GetEventsForCompaction returns the slice of events that should be compacted.
// These are events after LastCompactedIndex, excluding the KeepRecent events.
func GetEventsForCompaction(sess session.Session, lastCompactedIndex int, keepRecent int) []*session.Event {
	events := sess.Events()
	totalEvents := events.Len()

	if totalEvents == 0 {
		return nil
	}

	// Start from after last compacted index
	startIdx := lastCompactedIndex + 1
	if startIdx >= totalEvents {
		return nil
	}

	// End before the keepRecent events
	endIdx := totalEvents - keepRecent
	if endIdx <= startIdx {
		// Not enough events to compact while keeping recent ones
		return nil
	}

	result := make([]*session.Event, 0, endIdx-startIdx)
	for i := startIdx; i < endIdx; i++ {
		event := events.At(i)
		if event != nil {
			result = append(result, event)
		}
	}

	return result
}

// NewConfig creates a Config with the provided strategy and sensible defaults.
// Use functional options to customize.
func NewConfig(strategy Strategy, opts ...ConfigOption) (*Config, error) {
	cfg := &Config{
		Strategy:   strategy,
		KeepRecent: DefaultKeepRecent,
	}

	for _, opt := range opts {
		opt(cfg)
	}

	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	return cfg, nil
}

// ConfigOption is a functional option for Config.
type ConfigOption func(*Config)

// WithMaxEvents sets the MaxEvents trigger threshold.
func WithMaxEvents(n int) ConfigOption {
	return func(c *Config) {
		c.MaxEvents = n
	}
}

// WithMaxTokens sets the MaxTokens trigger threshold.
func WithMaxTokens(n int) ConfigOption {
	return func(c *Config) {
		c.MaxTokens = n
	}
}

// WithInterval sets the Interval trigger.
func WithInterval(n int) ConfigOption {
	return func(c *Config) {
		c.Interval = n
	}
}

// WithKeepRecent sets the number of recent events to preserve.
func WithKeepRecent(n int) ConfigOption {
	return func(c *Config) {
		c.KeepRecent = n
	}
}

// WithEnableAfterAgent enables post-run compaction check.
func WithEnableAfterAgent(enabled bool) ConfigOption {
	return func(c *Config) {
		c.EnableAfterAgent = enabled
	}
}

// WithStateKey sets a custom session.State key for compaction state.
func WithStateKey(key string) ConfigOption {
	return func(c *Config) {
		c.stateKey = key
	}
}

// SummaryEvent creates a system event containing a compaction summary.
// This is used to store the summary in the session history.
func SummaryEvent(summary string) *session.Event {
	return &session.Event{
		Author:    "system",
		Timestamp: time.Now(),
		LLMResponse: model.LLMResponse{
			Content: genai.NewContentFromText(
				"[Previous conversation summarized]: "+summary,
				genai.RoleModel,
			),
		},
	}
}

// ExtractSummaryText extracts the first text part from an LLM response.
// Returns empty string if response is nil or has no text content.
func ExtractSummaryText(response *model.LLMResponse) string {
	if response == nil || response.Content == nil {
		return ""
	}
	var summary string
	for _, part := range response.Content.Parts {
		if part != nil && part.Text != "" {
			summary += part.Text
		}
	}
	return summary
}
