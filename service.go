package memory

import (
	"context"
	"fmt"

	"github.com/ieshan/adk-go-memory/adapter"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// Compile-time interface compliance check
var _ memory.Service = (*Service)(nil)

// ServiceConfig configures the Service.
type ServiceConfig struct {
	Storage  adapter.Storage
	Deriver  *Deriver
	Provider *Provider
}

// Service implements google.golang.org/adk/memory.Service with automatic observation extraction and retrieval.
type Service struct {
	storage  adapter.Storage
	deriver  *Deriver
	provider *Provider
}

// NewService creates a new Service with the given configuration.
func NewService(cfg ServiceConfig) *Service {
	return &Service{
		storage:  cfg.Storage,
		deriver:  cfg.Deriver,
		provider: cfg.Provider,
	}
}

// AddSessionToMemory implements memory.Service.
// Extracts observations from session events and stores them.
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
			}
			// Author enrichment for future use
			_ = ev.Author
			messages = append(messages, msg)
		}
	}

	if len(messages) == 0 {
		return nil
	}

	return s.deriver.Derive(ctx, messages, sess.ID(), sess.UserID(), sess.AppName())
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
			ID: obs.ID,
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
