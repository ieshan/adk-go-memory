// Package memory provides ADK-Go memory service components.
//
// This file contains the Provider which wires together all memory components
// for comprehensive context assembly including representation management
// and peer cards.
package memory

import (
	"context"
	"fmt"

	"github.com/ieshan/adk-go-memory/adapter"
)

// Provider wires together all memory components for ADK-Go.
type Provider struct {
	storage       adapter.Storage
	repManager    *RepresentationManager
	peerCards     map[string]*PeerCard
	embeddingFunc func(ctx context.Context, text string) ([]float32, error)
}

// ProviderConfig configures the Provider.
type ProviderConfig struct {
	Storage       adapter.Storage
	EmbeddingFunc func(ctx context.Context, text string) ([]float32, error)
}

// NewProvider creates a new Provider with the given configuration.
func NewProvider(cfg ProviderConfig) *Provider {
	p := &Provider{
		storage:       cfg.Storage,
		peerCards:     make(map[string]*PeerCard),
		embeddingFunc: cfg.EmbeddingFunc,
	}

	// Initialize representation manager
	if cfg.Storage != nil {
		p.repManager = NewRepresentationManager(RepresentationConfig{
			Storage:           cfg.Storage,
			SemanticBudget:    5,
			MostDerivedBudget: 5,
			RecentBudget:      5,
			EmbeddingFunc:     cfg.EmbeddingFunc,
		})
	}

	return p
}

// GetMemoryContext retrieves memory context for the given query.
func (p *Provider) GetMemoryContext(ctx context.Context, query string, sessionID, userID, appName string) (string, error) {
	if p.storage == nil {
		return "", nil
	}

	// Use representation manager for comprehensive context
	if p.repManager != nil {
		rep, err := p.repManager.GetWorkingRepresentation(ctx, query, sessionID, userID, appName)
		if err == nil && len(rep.Observations) > 0 {
			return rep.Format(), nil
		}
	}

	// Fallback to simple search
	results, err := p.storage.Search(ctx, &adapter.SearchOptions{
		Query:      query,
		MaxResults: 10,
		Mode:       adapter.SearchModeHybrid,
		SessionID:  sessionID,
		UserID:     userID,
		AppName:    appName,
	})
	if err != nil {
		return "", fmt.Errorf("provider: search: %w", err)
	}

	var context string
	for _, r := range results {
		context += fmt.Sprintf("- %s\n", r.Observation.Content)
	}

	return context, nil
}

// SearchMemory provides the tool-accessible search interface.
// This is called by both MemoryTool.Run and PreloadMemoryTool.ProcessRequest.
func (p *Provider) SearchMemory(ctx context.Context, query, sessionID, userID, appName string) ([]adapter.Observation, error) {
	if p.storage == nil {
		return nil, nil
	}

	// Use representation manager for comprehensive context
	if p.repManager != nil && query != "" {
		rep, err := p.repManager.GetWorkingRepresentation(ctx, query, sessionID, userID, appName)
		if err == nil && len(rep.Observations) > 0 {
			return rep.Observations, nil
		}
	}

	// Fallback to simple search
	results, err := p.storage.Search(ctx, &adapter.SearchOptions{
		Query:      query,
		MaxResults: 10,
		Mode:       adapter.SearchModeHybrid,
		SessionID:  sessionID,
		UserID:     userID,
		AppName:    appName,
	})
	if err != nil {
		return nil, fmt.Errorf("provider: search: %w", err)
	}

	observations := make([]adapter.Observation, len(results))
	for i, r := range results {
		observations[i] = r.Observation
	}
	return observations, nil
}

// GetOrCreatePeerCard gets or creates a peer card for the given peer ID.
func (p *Provider) GetOrCreatePeerCard(peerID string) *PeerCard {
	if pc, ok := p.peerCards[peerID]; ok {
		return pc
	}
	pc := NewPeerCard(peerID)
	p.peerCards[peerID] = pc
	return pc
}

// LoadPeerCardFromMemory loads peer card facts from storage.
// Uses QueryMostDerived to retrieve the most important observations
// (highest derivation count = most referenced) for the peer,
// rather than just the most recent ones.
func (p *Provider) LoadPeerCardFromMemory(ctx context.Context, peerID string) error {
	if p.storage == nil {
		return nil
	}

	observations, err := p.storage.QueryMostDerived(ctx, "", peerID, "", maxPeerCardFacts)
	if err != nil {
		return fmt.Errorf("provider: failed to load peer card from memory: %w", err)
	}

	pc := p.GetOrCreatePeerCard(peerID)
	for _, obs := range observations {
		pc.AddFact(PeerFact{
			Content: obs.Content,
			Score:   obs.Score(),
			Type:    obs.Level,
			Tags:    obs.Tags,
		})
	}

	return nil
}

// OnSessionStart is called when a session starts.
func (p *Provider) OnSessionStart(ctx context.Context, sessionID, userID, appName string) error {
	// Load peer card if user is known
	if userID != "" {
		if err := p.LoadPeerCardFromMemory(ctx, userID); err != nil {
			return err
		}
	}
	return nil
}

// Close releases resources held by the provider.
func (p *Provider) Close() error {
	if p.storage != nil {
		return p.storage.Close()
	}
	return nil
}
