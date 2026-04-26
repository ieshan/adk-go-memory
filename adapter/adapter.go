// Package adapter provides storage abstractions for the memory layer.
package adapter

import (
	"context"
	"math"
	"time"
)

// ObservationLevel indicates the confidence level of an observation.
type ObservationLevel string

const (
	// LevelExplicit is for directly stated facts.
	LevelExplicit ObservationLevel = "explicit"
	// LevelDeductive is for logical inferences from stated facts.
	LevelDeductive ObservationLevel = "deductive"
	// LevelInductive is for patterns inferred from multiple observations.
	LevelInductive ObservationLevel = "inductive"
	// LevelContradiction is for conflicting statements.
	LevelContradiction ObservationLevel = "contradiction"
)

// LevelBaseScore returns the base score for an observation level.
// Higher scores indicate higher confidence.
func LevelBaseScore(level ObservationLevel) float64 {
	switch level {
	case LevelExplicit:
		return 0.9
	case LevelDeductive:
		return 0.7
	case LevelInductive:
		return 0.5
	case LevelContradiction:
		return 0.3
	default:
		return 0.5
	}
}

// Observation represents a single extracted fact.
type Observation struct {
	ID           string
	Content      string
	Level        ObservationLevel
	SessionID    string
	UserID       string
	AppName      string
	Tags         []string
	TimesDerived int
	CreatedAt    time.Time
	Embedding    []float32
}

// Score returns a composite score based on level and times_derived.
func (o Observation) Score() float64 {
	base := LevelBaseScore(o.Level)
	// Boost by times_derived with diminishing returns
	// Using logarithmic scaling: log2(derived+1) * 0.1, capped at 0.3
	boost := math.Min(math.Log2(float64(o.TimesDerived+1))*0.1, 0.3)
	return math.Min(base+boost, 1.0)
}

// SearchMode indicates which search strategy to use.
type SearchMode int

const (
	// SearchModeHybrid uses reciprocal rank fusion of both vector and FTS.
	// This is the zero value so that uninitialized SearchOptions default to
	// the most comprehensive search strategy.
	SearchModeHybrid SearchMode = iota
	// SearchModeVector uses vector similarity search.
	SearchModeVector
	// SearchModeFTS uses full-text search.
	SearchModeFTS
)

// SearchOptions configures a search operation.
type SearchOptions struct {
	Query      string
	Embedding  []float32
	MaxResults int
	Mode       SearchMode
	SessionID  string
	UserID     string
	AppName    string
}

// SearchResult is an observation with a relevance score.
type SearchResult struct {
	Observation Observation
	Score       float64
	Source      string // "vector", "fts", "rrf"
}

// Storage defines the interface for observation storage backends.
type Storage interface {
	// Store saves an observation to storage.
	Store(ctx context.Context, obs *Observation) error

	// GetByID retrieves an observation by its ID.
	GetByID(ctx context.Context, id string) (*Observation, error)

	// Search finds observations matching the given options.
	Search(ctx context.Context, opts *SearchOptions) ([]SearchResult, error)

	// Forget deletes an observation by ID.
	Forget(ctx context.Context, id string) error

	// Purge deletes observations matching the filter.
	Purge(ctx context.Context, filter map[string]string) error

	// IncrementTimesDerived increments the times_derived counter for an observation.
	IncrementTimesDerived(ctx context.Context, id string) error

	// Close releases any resources held by the storage.
	Close() error

	// QueryMostDerived returns observations sorted by times_derived DESC (most referenced first).
	// Used by RepresentationManager for building working representations.
	QueryMostDerived(ctx context.Context, sessionID, userID, appName string, limit int) ([]Observation, error)

	// QueryRecent returns observations sorted by created_at DESC (most recent first).
	// Used by RepresentationManager for building working representations.
	QueryRecent(ctx context.Context, sessionID, userID, appName string, limit int) ([]Observation, error)
}
