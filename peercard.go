package memory

import (
	"cmp"
	"fmt"
	"slices"
	"strings"

	"github.com/ieshan/adk-go-memory/adapter"
)

// Maximum facts per peer card is 40.
const maxPeerCardFacts = 40

// PeerFact represents a single fact about a peer.
type PeerFact struct {
	Content string
	Score   float64
	Type    adapter.ObservationLevel
	Tags    []string
}

// PeerCard stores facts about a peer (user).
type PeerCard struct {
	peerID string
	facts  []PeerFact
}

// NewPeerCard creates a new peer card for the given peer ID.
func NewPeerCard(peerID string) *PeerCard {
	return &PeerCard{
		peerID: peerID,
		facts:  make([]PeerFact, 0),
	}
}

// PeerID returns the ID of the peer this card represents.
func (pc *PeerCard) PeerID() string {
	return pc.peerID
}

// AddFact adds a fact to the peer card.
// If the card is at capacity, the lowest-scoring fact is evicted only if
// the new fact has a higher score; otherwise the new fact is discarded.
func (pc *PeerCard) AddFact(fact PeerFact) {
	if len(pc.facts) >= maxPeerCardFacts {
		// Find lowest scoring fact
		lowestIdx := 0
		for i, f := range pc.facts {
			if f.Score < pc.facts[lowestIdx].Score {
				lowestIdx = i
			}
		}
		// Only evict if the new fact scores higher than the worst existing one
		if fact.Score <= pc.facts[lowestIdx].Score {
			return
		}
		pc.facts = append(pc.facts[:lowestIdx], pc.facts[lowestIdx+1:]...)
	}
	pc.facts = append(pc.facts, fact)
}

// Facts returns all facts in the peer card.
func (pc *PeerCard) Facts() []PeerFact {
	return pc.facts
}

// ReplaceFacts replaces all facts with the given slice.
func (pc *PeerCard) ReplaceFacts(facts []PeerFact) {
	pc.facts = facts
}

// Prune removes facts with scores below the threshold.
func (pc *PeerCard) Prune(threshold float64) {
	var pruned []PeerFact
	for _, f := range pc.facts {
		if f.Score >= threshold {
			pruned = append(pruned, f)
		}
	}
	pc.facts = pruned
}

// ByType returns facts filtered by observation level.
func (pc *PeerCard) ByType(level adapter.ObservationLevel) []PeerFact {
	var filtered []PeerFact
	for _, f := range pc.facts {
		if f.Type == level {
			filtered = append(filtered, f)
		}
	}
	return filtered
}

// ByTag returns facts that have the given tag.
func (pc *PeerCard) ByTag(tag string) []PeerFact {
	var filtered []PeerFact
	for _, f := range pc.facts {
		for _, t := range f.Tags {
			if t == tag {
				filtered = append(filtered, f)
				break
			}
		}
	}
	return filtered
}

// Render returns a formatted string representation of the peer card.
func (pc *PeerCard) Render() string {
	if len(pc.facts) == 0 {
		return fmt.Sprintf("[Peer card for %s: no facts recorded]", pc.peerID)
	}

	// Sort a copy by score descending to avoid mutating internal state
	sorted := make([]PeerFact, len(pc.facts))
	copy(sorted, pc.facts)
	slices.SortFunc(sorted, func(a, b PeerFact) int {
		return cmp.Compare(b.Score, a.Score)
	})

	// Group by type
	byType := make(map[adapter.ObservationLevel][]PeerFact)
	for _, f := range sorted {
		byType[f.Type] = append(byType[f.Type], f)
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("[Peer card for %s]\n", pc.peerID))

	for _, level := range []adapter.ObservationLevel{
		adapter.LevelExplicit,
		adapter.LevelDeductive,
		adapter.LevelInductive,
		adapter.LevelContradiction,
	} {
		if facts, ok := byType[level]; ok && len(facts) > 0 {
			sb.WriteString(fmt.Sprintf("[%s]\n", level))
			for _, f := range facts {
				sb.WriteString(fmt.Sprintf("  - %s (score: %.2f)\n", f.Content, f.Score))
			}
		}
	}

	return sb.String()
}
