package adapter

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/ieshan/idx"
)

// Compile-time interface compliance check.
var _ Storage = (*MemoryStorage)(nil)

// MemoryStorage implements Storage using an in-memory map.
// This is a lightweight implementation suitable for testing and demos.
// It does NOT support true vector or FTS search; Search uses simple substring matching.
type MemoryStorage struct {
	mu           sync.RWMutex
	observations map[idx.ID]*Observation
}

// InMemory creates a lightweight map-based Storage for tests and demos.
// It does NOT support vector/FTS search; Search falls back to simple text matching.
func InMemory() *MemoryStorage {
	return &MemoryStorage{
		observations: make(map[idx.ID]*Observation),
	}
}

// Store saves an observation to storage.
func (s *MemoryStorage) Store(ctx context.Context, obs *Observation) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.observations[obs.ID]; exists {
		return fmt.Errorf("memory: observation with ID %s already exists", obs.ID)
	}

	// Clone the observation to avoid external mutations
	cloned := &Observation{
		ID:           obs.ID,
		Content:      obs.Content,
		Level:        obs.Level,
		SessionID:    obs.SessionID,
		UserID:       obs.UserID,
		AppName:      obs.AppName,
		Tags:         append([]string(nil), obs.Tags...),
		TimesDerived: obs.TimesDerived,
		CreatedAt:    obs.CreatedAt,
	}
	if len(obs.Embedding) > 0 {
		cloned.Embedding = make([]float32, len(obs.Embedding))
		copy(cloned.Embedding, obs.Embedding)
	}

	s.observations[obs.ID] = cloned
	return nil
}

// GetByID retrieves an observation by its ID.
func (s *MemoryStorage) GetByID(ctx context.Context, id idx.ID) (*Observation, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	obs, exists := s.observations[id]
	if !exists {
		return nil, fmt.Errorf("observation not found: %s", id.String())
	}

	// Return a clone to avoid external mutation
	return cloneObservation(obs), nil
}

// Search finds observations matching the given options.
// For MemoryStorage, this uses simple substring matching on the content.
// Vector search mode returns empty results since MemoryStorage doesn't support embeddings.
func (s *MemoryStorage) Search(ctx context.Context, opts *SearchOptions) ([]SearchResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	maxResults := opts.MaxResults
	if maxResults == 0 {
		maxResults = 10
	}

	// MemoryStorage doesn't support vector search; return empty for vector-only mode
	if opts.Mode == SearchModeVector && opts.Query == "" {
		return []SearchResult{}, nil
	}

	var results []SearchResult
	for _, obs := range s.observations {
		// Apply filters
		if opts.SessionID != "" && obs.SessionID != opts.SessionID {
			continue
		}
		if opts.UserID != "" && obs.UserID != opts.UserID {
			continue
		}
		if opts.AppName != "" && obs.AppName != opts.AppName {
			continue
		}

		// Simple substring match for text search
		if opts.Query != "" && !strings.Contains(strings.ToLower(obs.Content), strings.ToLower(opts.Query)) {
			continue
		}

		results = append(results, SearchResult{
			Observation: *cloneObservation(obs),
			Score:       obs.Score(),
			Source:      "fts", // MemoryStorage uses text matching, closest to FTS
		})

		if len(results) >= maxResults {
			break
		}
	}

	return results, nil
}

// Forget deletes an observation by ID.
func (s *MemoryStorage) Forget(ctx context.Context, id idx.ID) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.observations, id)
	return nil
}

// Purge deletes observations matching the filter.
func (s *MemoryStorage) Purge(ctx context.Context, filter map[string]string) error {
	validKeys := map[string]bool{"session_id": true, "user_id": true, "app_name": true}
	for key := range filter {
		if !validKeys[key] {
			return fmt.Errorf("memory: purge: unrecognized filter key %q (valid keys: session_id, user_id, app_name)", key)
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Build filter check
	for id, obs := range s.observations {
		match := true
		if sessionID, ok := filter["session_id"]; ok && obs.SessionID != sessionID {
			match = false
		}
		if userID, ok := filter["user_id"]; ok && obs.UserID != userID {
			match = false
		}
		if appName, ok := filter["app_name"]; ok && obs.AppName != appName {
			match = false
		}

		if match {
			delete(s.observations, id)
		}
	}

	return nil
}

// IncrementTimesDerived increments the times_derived counter for an observation.
func (s *MemoryStorage) IncrementTimesDerived(ctx context.Context, id idx.ID) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	obs, exists := s.observations[id]
	if !exists {
		return fmt.Errorf("memory: increment times_derived: observation not found: %s", id.String())
	}

	obs.TimesDerived++
	return nil
}

// Close is a no-op for MemoryStorage.
func (s *MemoryStorage) Close() error {
	return nil
}

// QueryMostDerived returns observations sorted by times_derived DESC.
func (s *MemoryStorage) QueryMostDerived(ctx context.Context, sessionID, userID, appName string, limit int) ([]Observation, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var filtered []*Observation
	for _, obs := range s.observations {
		if sessionID != "" && obs.SessionID != sessionID {
			continue
		}
		if userID != "" && obs.UserID != userID {
			continue
		}
		if appName != "" && obs.AppName != appName {
			continue
		}
		filtered = append(filtered, obs)
	}

	// Sort by TimesDerived DESC, then CreatedAt DESC
	for i := 0; i < len(filtered); i++ {
		for j := i + 1; j < len(filtered); j++ {
			if filtered[j].TimesDerived > filtered[i].TimesDerived {
				filtered[i], filtered[j] = filtered[j], filtered[i]
			} else if filtered[j].TimesDerived == filtered[i].TimesDerived &&
				filtered[j].CreatedAt.After(filtered[i].CreatedAt) {
				filtered[i], filtered[j] = filtered[j], filtered[i]
			}
		}
	}

	if limit > len(filtered) {
		limit = len(filtered)
	}

	result := make([]Observation, limit)
	for i := 0; i < limit; i++ {
		result[i] = *cloneObservation(filtered[i])
	}
	return result, nil
}

// QueryRecent returns observations sorted by created_at DESC.
func (s *MemoryStorage) QueryRecent(ctx context.Context, sessionID, userID, appName string, limit int) ([]Observation, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var filtered []*Observation
	for _, obs := range s.observations {
		if sessionID != "" && obs.SessionID != sessionID {
			continue
		}
		if userID != "" && obs.UserID != userID {
			continue
		}
		if appName != "" && obs.AppName != appName {
			continue
		}
		filtered = append(filtered, obs)
	}

	// Sort by CreatedAt DESC
	for i := 0; i < len(filtered); i++ {
		for j := i + 1; j < len(filtered); j++ {
			if filtered[j].CreatedAt.After(filtered[i].CreatedAt) {
				filtered[i], filtered[j] = filtered[j], filtered[i]
			}
		}
	}

	if limit > len(filtered) {
		limit = len(filtered)
	}

	result := make([]Observation, limit)
	for i := 0; i < limit; i++ {
		result[i] = *cloneObservation(filtered[i])
	}
	return result, nil
}

// cloneObservation creates a deep copy of an observation.
func cloneObservation(obs *Observation) *Observation {
	cloned := &Observation{
		ID:           obs.ID,
		Content:      obs.Content,
		Level:        obs.Level,
		SessionID:    obs.SessionID,
		UserID:       obs.UserID,
		AppName:      obs.AppName,
		Tags:         append([]string(nil), obs.Tags...),
		TimesDerived: obs.TimesDerived,
		CreatedAt:    obs.CreatedAt,
	}
	if len(obs.Embedding) > 0 {
		cloned.Embedding = make([]float32, len(obs.Embedding))
		copy(cloned.Embedding, obs.Embedding)
	}
	return cloned
}
