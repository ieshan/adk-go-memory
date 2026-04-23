package memory

import (
	"fmt"
	"strings"
	"testing"

	"github.com/ieshan/adk-go-memory/adapter"
)

func TestNewPeerCard(t *testing.T) {
	pc := NewPeerCard("user-1")
	if pc.PeerID() != "user-1" {
		t.Errorf("PeerID() = %v, want user-1", pc.PeerID())
	}
}

func TestPeerCard_AddFact(t *testing.T) {
	pc := NewPeerCard("user-1")
	pc.AddFact(PeerFact{Content: "likes Go", Score: 0.9, Type: adapter.LevelExplicit})

	facts := pc.Facts()
	if len(facts) != 1 {
		t.Errorf("len(Facts()) = %d, want 1", len(facts))
	}
	if facts[0].Content != "likes Go" {
		t.Errorf("Content = %v, want 'likes Go'", facts[0].Content)
	}
}

func TestPeerCard_AddFact_MaxLimit(t *testing.T) {
	pc := NewPeerCard("user-1")

	// Add max + 1 facts
	for i := 0; i <= maxPeerCardFacts; i++ {
		pc.AddFact(PeerFact{Content: "fact", Score: 0.5, Type: adapter.LevelExplicit})
	}

	facts := pc.Facts()
	if len(facts) != maxPeerCardFacts {
		t.Errorf("len(Facts()) = %d, want %d", len(facts), maxPeerCardFacts)
	}
}

func TestPeerCard_ReplaceFacts(t *testing.T) {
	pc := NewPeerCard("user-1")
	pc.AddFact(PeerFact{Content: "old fact", Score: 0.5, Type: adapter.LevelExplicit})

	newFacts := []PeerFact{
		{Content: "fact 1", Score: 0.9, Type: adapter.LevelExplicit},
		{Content: "fact 2", Score: 0.8, Type: adapter.LevelDeductive},
	}
	pc.ReplaceFacts(newFacts)

	facts := pc.Facts()
	if len(facts) != 2 {
		t.Errorf("len(Facts()) = %d, want 2", len(facts))
	}
	if facts[0].Content != "fact 1" {
		t.Errorf("First fact = %v, want 'fact 1'", facts[0].Content)
	}
}

func TestPeerCard_Prune(t *testing.T) {
	pc := NewPeerCard("user-1")
	pc.AddFact(PeerFact{Content: "high score", Score: 0.9, Type: adapter.LevelExplicit})
	pc.AddFact(PeerFact{Content: "low score", Score: 0.2, Type: adapter.LevelContradiction})
	pc.AddFact(PeerFact{Content: "medium score", Score: 0.5, Type: adapter.LevelExplicit})

	pc.Prune(0.4)

	facts := pc.Facts()
	if len(facts) != 2 {
		t.Errorf("len(Facts()) = %d, want 2", len(facts))
	}

	for _, f := range facts {
		if f.Score < 0.4 {
			t.Errorf("Fact with score %v should have been pruned", f.Score)
		}
	}
}

func TestPeerCard_ByType(t *testing.T) {
	pc := NewPeerCard("user-1")
	pc.AddFact(PeerFact{Content: "explicit fact", Score: 0.9, Type: adapter.LevelExplicit})
	pc.AddFact(PeerFact{Content: "deductive fact", Score: 0.8, Type: adapter.LevelDeductive})
	pc.AddFact(PeerFact{Content: "another explicit", Score: 0.7, Type: adapter.LevelExplicit})

	explicitFacts := pc.ByType(adapter.LevelExplicit)
	if len(explicitFacts) != 2 {
		t.Errorf("len(ByType(explicit)) = %d, want 2", len(explicitFacts))
	}

	deductiveFacts := pc.ByType(adapter.LevelDeductive)
	if len(deductiveFacts) != 1 {
		t.Errorf("len(ByType(deductive)) = %d, want 1", len(deductiveFacts))
	}
}

func TestPeerCard_ByTag(t *testing.T) {
	pc := NewPeerCard("user-1")
	pc.AddFact(PeerFact{Content: "fact 1", Score: 0.9, Type: adapter.LevelExplicit, Tags: []string{"work", "coding"}})
	pc.AddFact(PeerFact{Content: "fact 2", Score: 0.8, Type: adapter.LevelExplicit, Tags: []string{"personal"}})
	pc.AddFact(PeerFact{Content: "fact 3", Score: 0.7, Type: adapter.LevelExplicit, Tags: []string{"work"}})

	workFacts := pc.ByTag("work")
	if len(workFacts) != 2 {
		t.Errorf("len(ByTag(work)) = %d, want 2", len(workFacts))
	}
}

func TestPeerCard_Render(t *testing.T) {
	pc := NewPeerCard("alice")
	pc.AddFact(PeerFact{Content: "prefers coffee", Score: 0.9, Type: adapter.LevelExplicit})
	pc.AddFact(PeerFact{Content: "works remotely", Score: 0.8, Type: adapter.LevelDeductive})

	rendered := pc.Render()

	if !strings.Contains(rendered, "[Peer card for alice]") {
		t.Error("Expected peer card header")
	}
	if !strings.Contains(rendered, "prefers coffee") {
		t.Error("Expected 'prefers coffee' in rendered output")
	}
	if !strings.Contains(rendered, "works remotely") {
		t.Error("Expected 'works remotely' in rendered output")
	}
	if !strings.Contains(rendered, "[explicit]") {
		t.Error("Expected [explicit] section")
	}
	if !strings.Contains(rendered, "[deductive]") {
		t.Error("Expected [deductive] section")
	}
}

func TestPeerCard_Render_Empty(t *testing.T) {
	pc := NewPeerCard("bob")
	rendered := pc.Render()

	expected := "[Peer card for bob: no facts recorded]"
	if rendered != expected {
		t.Errorf("Render() = %v, want %v", rendered, expected)
	}
}

func TestPeerCard_EvictionDoesNotReplaceHigherScore(t *testing.T) {
	pc := NewPeerCard("user-1")

	// Fill to max capacity with high-scoring facts
	for i := 0; i < maxPeerCardFacts; i++ {
		pc.AddFact(PeerFact{
			Content: fmt.Sprintf("high-scoring fact %d", i),
			Score:   0.9,
			Type:    adapter.LevelExplicit,
		})
	}

	factsBefore := len(pc.Facts())

	// Try to add a low-scoring fact - should be discarded
	pc.AddFact(PeerFact{
		Content: "low scoring fact",
		Score:   0.3,
		Type:    adapter.LevelContradiction,
	})

	factsAfter := len(pc.Facts())
	if factsAfter != factsBefore {
		t.Errorf("len(Facts) = %d, want %d (low-scoring fact should be discarded)", factsAfter, factsBefore)
	}

	// Verify none of the existing facts were replaced
	for _, f := range pc.Facts() {
		if f.Content == "low scoring fact" {
			t.Error("Low-scoring fact should not have been added")
		}
		if f.Score < 0.9 {
			t.Errorf("Found fact with score %f, all should be >= 0.9", f.Score)
		}
	}
}

func TestPeerCard_EvictionReplacesLowestScore(t *testing.T) {
	pc := NewPeerCard("user-1")

	// Fill to max capacity with varying scores
	for i := 0; i < maxPeerCardFacts; i++ {
		score := 0.3 + float64(i)*0.02 // scores from 0.3 to ~1.08
		pc.AddFact(PeerFact{
			Content: fmt.Sprintf("fact %d", i),
			Score:   score,
			Type:    adapter.LevelExplicit,
		})
	}

	// Add a high-scoring fact that should evict the lowest
	pc.AddFact(PeerFact{
		Content: "new high scoring fact",
		Score:   0.95,
		Type:    adapter.LevelExplicit,
	})

	found := false
	for _, f := range pc.Facts() {
		if f.Content == "new high scoring fact" {
			found = true
			break
		}
	}
	if !found {
		t.Error("High-scoring fact should have been added, evicting the lowest")
	}
}

func TestPeerCard_Render_DoesNotMutateFacts(t *testing.T) {
	pc := NewPeerCard("user-1")
	pc.AddFact(PeerFact{Content: "low score fact", Score: 0.3, Type: adapter.LevelContradiction})
	pc.AddFact(PeerFact{Content: "high score fact", Score: 0.9, Type: adapter.LevelExplicit})

	// Capture facts before Render
	factsBefore := make([]PeerFact, len(pc.Facts()))
	copy(factsBefore, pc.Facts())

	// Call Render (which sorts internally)
	_ = pc.Render()

	// Verify facts order is unchanged
	factsAfter := pc.Facts()
	if len(factsAfter) != len(factsBefore) {
		t.Fatalf("Facts count changed after Render: got %d, want %d", len(factsAfter), len(factsBefore))
	}
	for i := range factsBefore {
		if factsAfter[i].Content != factsBefore[i].Content || factsAfter[i].Score != factsBefore[i].Score {
			t.Errorf("Facts[%d] changed after Render: got {%s, %.1f}, want {%s, %.1f}",
				i, factsAfter[i].Content, factsAfter[i].Score,
				factsBefore[i].Content, factsBefore[i].Score)
		}
	}
}

func TestPeerCard_Prune_AllBelowThreshold(t *testing.T) {
	pc := NewPeerCard("user-1")
	pc.AddFact(PeerFact{Content: "low 1", Score: 0.1, Type: adapter.LevelContradiction})
	pc.AddFact(PeerFact{Content: "low 2", Score: 0.2, Type: adapter.LevelContradiction})

	pc.Prune(0.5)

	facts := pc.Facts()
	if len(facts) != 0 {
		t.Errorf("Expected 0 facts after pruning all, got %d", len(facts))
	}
}

func TestPeerCard_Prune_NoneBelowThreshold(t *testing.T) {
	pc := NewPeerCard("user-1")
	pc.AddFact(PeerFact{Content: "high 1", Score: 0.9, Type: adapter.LevelExplicit})
	pc.AddFact(PeerFact{Content: "high 2", Score: 0.8, Type: adapter.LevelExplicit})

	pc.Prune(0.5)

	facts := pc.Facts()
	if len(facts) != 2 {
		t.Errorf("Expected 2 facts after pruning none, got %d", len(facts))
	}
}

func TestPeerCard_ReplaceFacts_RespectsMaxOnNextAdd(t *testing.T) {
	pc := NewPeerCard("user-1")

	// Replace with more than max facts
	var facts []PeerFact
	for i := 0; i < maxPeerCardFacts+10; i++ {
		facts = append(facts, PeerFact{Content: fmt.Sprintf("fact %d", i), Score: 0.5, Type: adapter.LevelExplicit})
	}
	pc.ReplaceFacts(facts)

	// ReplaceFacts bypasses the max limit, but next AddFact should still enforce it
	pc.AddFact(PeerFact{Content: "overflow fact", Score: 0.3, Type: adapter.LevelContradiction})

	// The card should have maxPeerCardFacts + 10 facts from ReplaceFacts,
	// plus the overflow fact only if it evicts a lower-scoring one.
	// Since all existing facts have score 0.5 and the new one is 0.3, it should be discarded.
	totalFacts := len(pc.Facts())
	if totalFacts != maxPeerCardFacts+10 {
		t.Errorf("Expected %d facts, got %d (low-scoring fact should be discarded)", maxPeerCardFacts+10, totalFacts)
	}
}
