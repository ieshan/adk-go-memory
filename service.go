// Package memory provides ADK-Go memory service with automatic
// observation extraction, semantic search, and context assembly.
package memory

import (
	"context"
	"fmt"

	"github.com/ieshan/adk-go-memory/adapter"
	"github.com/ieshan/adk-go-memory/compaction"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// Compile-time interface compliance check
var _ memory.Service = (*Service)(nil)

// ServiceConfig configures the Service.
type ServiceConfig struct {
	Storage            adapter.Storage
	Deriver            *Deriver
	Provider           *Provider
	EnableDeltaMode    bool   // NEW: use NumRecentEvents for delta processing
	CompactionStateKey string // NEW: session.State key for compaction tracking
}

// Service implements google.golang.org/adk/memory.Service with automatic observation extraction and retrieval.
type Service struct {
	storage  adapter.Storage
	deriver  *Deriver
	provider *Provider
	config   ServiceConfig
}

// NewService creates a new Service with the given configuration.
func NewService(cfg ServiceConfig) *Service {
	return &Service{
		storage:  cfg.Storage,
		deriver:  cfg.Deriver,
		provider: cfg.Provider,
		config:   cfg,
	}
}

// AddSessionToMemory implements memory.Service.
// Extracts observations from session events and stores them.
// When EnableDeltaMode is true, only processes events since last compaction.
func (s *Service) AddSessionToMemory(ctx context.Context, sess session.Session) error {
	if s.deriver == nil {
		return nil
	}

	if sess == nil {
		return nil
	}

	events := sess.Events()
	if events == nil {
		return nil
	}

	var messages []TimestampedMessage

	if s.config.EnableDeltaMode {
		// Delta mode: only process events since last compaction
		lastIndex := s.getLastCompactedIndex(sess)
		messages = s.getEventsSince(sess, lastIndex)
	} else {
		// Original: process all events
		for ev := range events.All() {
			if ev == nil || ev.Content == nil {
				continue
			}

			// Extract text content from genai.Content
			var textContent string
			for _, part := range ev.Content.Parts {
				if part.Text != "" {
					textContent += part.Text + " "
				}
			}

			if textContent != "" {
				msg := TimestampedMessage{
					Content: ev.Content,
					At:      ev.Timestamp,
					Author:  ev.Author,
				}
				messages = append(messages, msg)
			}
		}
	}

	if len(messages) == 0 {
		return nil
	}

	err := s.deriver.Derive(ctx, messages, sess.ID(), sess.UserID(), sess.AppName())

	// After successful observation extraction, update LastEventCount
	// This ensures Interval trigger compares against post-compaction state
	if err == nil && s.config.EnableDeltaMode && s.config.CompactionStateKey != "" {
		if st, err := compaction.GetCompactionState(sess, s.config.CompactionStateKey); err == nil {
			st.LastEventCount = sess.Events().Len()
			_ = compaction.SaveCompactionState(sess, s.config.CompactionStateKey, st)
		}
	}

	return err
}

// getLastCompactedIndex retrieves the index of the last event included in a compaction summary.
// Returns -1 if no compaction has occurred.
func (s *Service) getLastCompactedIndex(sess session.Session) int {
	key := s.config.CompactionStateKey
	if key == "" {
		key = "adk_memory_compaction_state"
	}

	value, err := sess.State().Get(key)
	if err != nil {
		// State key not found - no compaction yet
		return -1
	}

	// Try to extract LastCompactedIndex from state struct
	// The state is expected to be *compaction.state or have a LastCompactedIndex field
	type compactionState interface {
		GetLastCompactedIndex() int
	}

	if cs, ok := value.(compactionState); ok {
		return cs.GetLastCompactedIndex()
	}

	// Fallback: try direct field access via reflection-like interface map
	if m, ok := value.(map[string]interface{}); ok {
		if idx, ok := m["last_compacted_index"].(int); ok {
			return idx
		}
	}

	return -1
}

// getEventsSince returns events after the given index (exclusive).
// If startIndex is -1, returns all events.
func (s *Service) getEventsSince(sess session.Session, startIndex int) []TimestampedMessage {
	events := sess.Events()
	if events == nil {
		return nil
	}

	var messages []TimestampedMessage
	idx := 0

	for ev := range events.All() {
		if ev == nil || ev.Content == nil {
			idx++
			continue
		}

		// Skip events up to and including startIndex
		if startIndex >= 0 && idx <= startIndex {
			idx++
			continue
		}

		// Extract text content from genai.Content
		var textContent string
		for _, part := range ev.Content.Parts {
			if part.Text != "" {
				textContent += part.Text + " "
			}
		}

		if textContent != "" {
			msg := TimestampedMessage{
				Content: ev.Content,
				At:      ev.Timestamp,
				Author:  ev.Author,
			}
			messages = append(messages, msg)
		}
		idx++
	}

	return messages
}

// SearchMemory implements memory.Service.
// Uses Provider for consistent behavior with tools (includes representation manager).
func (s *Service) SearchMemory(ctx context.Context, req *memory.SearchRequest) (*memory.SearchResponse, error) {
	if s.provider == nil {
		return &memory.SearchResponse{Memories: nil}, nil
	}

	// Use provider for comprehensive search (includes representation manager)
	observations, err := s.provider.SearchMemory(ctx, req.Query, "", req.UserID, req.AppName)
	if err != nil {
		return nil, fmt.Errorf("service: search memory: %w", err)
	}

	memories := make([]memory.Entry, len(observations))
	for i, obs := range observations {
		memories[i] = memory.Entry{
			ID: obs.ID.String(),
			Content: &genai.Content{
				Role:  "memory",
				Parts: []*genai.Part{{Text: obs.Content}},
			},
			Timestamp: obs.CreatedAt,
			CustomMetadata: map[string]any{
				"level": obs.Level,
				"tags":  obs.Tags,
				"score": obs.Score(),
			},
		}
	}

	return &memory.SearchResponse{Memories: memories}, nil
}

// Close releases resources held by the service.
func (s *Service) Close() error {
	if s.storage != nil {
		return s.storage.Close()
	}
	return nil
}
