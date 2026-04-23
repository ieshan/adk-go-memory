package memory

import (
	"context"
	"fmt"

	"github.com/ieshan/adk-go-memory/adapter"
)

// RepresentationConfig configures the RepresentationManager.
type RepresentationConfig struct {
	Storage           adapter.Storage
	SemanticBudget    int
	MostDerivedBudget int
	RecentBudget      int
	EmbeddingFunc     func(ctx context.Context, text string) ([]float32, error)
}

// WorkingRepresentation holds observations assembled for context injection.
type WorkingRepresentation struct {
	Observations  []adapter.Observation
	SemanticCount int
	DerivedCount  int
	RecentCount   int
	TotalScore    float64
}

// RepresentationManager retrieves working representations using multiple strategies for context assembly.
type RepresentationManager struct {
	storage adapter.Storage
	config  RepresentationConfig
}

// NewRepresentationManager creates a new RepresentationManager.
func NewRepresentationManager(cfg RepresentationConfig) *RepresentationManager {
	return &RepresentationManager{
		storage: cfg.Storage,
		config:  cfg,
	}
}

// GetWorkingRepresentation retrieves a working representation for the given query.
func (rm *RepresentationManager) GetWorkingRepresentation(ctx context.Context, query, sessionID, userID, appName string) (*WorkingRepresentation, error) {
	rep := &WorkingRepresentation{
		Observations: make([]adapter.Observation, 0),
	}

	// 1. Semantic search (if embedding function provided)
	if rm.config.EmbeddingFunc != nil && rm.config.SemanticBudget > 0 {
		embedding, err := rm.config.EmbeddingFunc(ctx, query)
		if err == nil && embedding != nil {
			results, err := rm.storage.Search(ctx, &adapter.SearchOptions{
				Embedding:  embedding,
				MaxResults: rm.config.SemanticBudget,
				Mode:       adapter.SearchModeVector,
				SessionID:  sessionID,
				UserID:     userID,
				AppName:    appName,
			})
			if err == nil {
				for _, r := range results {
					rep.Observations = append(rep.Observations, r.Observation)
					rep.SemanticCount++
					rep.TotalScore += r.Score
				}
			}
		}
	}

	// 2. Most derived observations (most referenced)
	if rm.config.MostDerivedBudget > 0 {
		derived, err := rm.storage.QueryMostDerived(ctx, sessionID, userID, appName, rm.config.MostDerivedBudget)
		if err == nil {
			for _, obs := range derived {
				if !containsObservation(rep.Observations, obs.ID) {
					rep.Observations = append(rep.Observations, obs)
					rep.DerivedCount++
					rep.TotalScore += obs.Score()
				}
			}
		}
	}

	// 3. Recent observations
	if rm.config.RecentBudget > 0 {
		recent, err := rm.storage.QueryRecent(ctx, sessionID, userID, appName, rm.config.RecentBudget)
		if err == nil {
			for _, obs := range recent {
				if !containsObservation(rep.Observations, obs.ID) {
					rep.Observations = append(rep.Observations, obs)
					rep.RecentCount++
					rep.TotalScore += obs.Score()
				}
			}
		}
	}

	return rep, nil
}

// containsObservation checks if an observation with the given ID is already in the list.
func containsObservation(obs []adapter.Observation, id string) bool {
	for _, o := range obs {
		if o.ID == id {
			return true
		}
	}
	return false
}

// Format returns a formatted string of the working representation.
func (wr *WorkingRepresentation) Format() string {
	var result string
	for i, obs := range wr.Observations {
		if i < wr.SemanticCount {
			result += fmt.Sprintf("[semantic] %s\n", obs.Content)
		} else if i < wr.SemanticCount+wr.DerivedCount {
			result += fmt.Sprintf("[derived] %s\n", obs.Content)
		} else {
			result += fmt.Sprintf("[recent] %s\n", obs.Content)
		}
	}
	return result
}
