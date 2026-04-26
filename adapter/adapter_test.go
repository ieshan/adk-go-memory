package adapter

import (
	"math"
	"testing"
)

func TestLevelBaseScore(t *testing.T) {
	tests := []struct {
		name     string
		level    ObservationLevel
		expected float64
	}{
		{"explicit", LevelExplicit, 0.9},
		{"deductive", LevelDeductive, 0.7},
		{"inductive", LevelInductive, 0.5},
		{"contradiction", LevelContradiction, 0.3},
		{"unknown", ObservationLevel("unknown"), 0.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := LevelBaseScore(tt.level)
			if got != tt.expected {
				t.Errorf("LevelBaseScore(%q) = %v, want %v", tt.level, got, tt.expected)
			}
		})
	}
}

func TestObservation_Score(t *testing.T) {
	tests := []struct {
		name         string
		level        ObservationLevel
		timesDerived int
		wantMin      float64
		wantMax      float64
	}{
		{
			name:         "explicit no boost",
			level:        LevelExplicit,
			timesDerived: 0,
			wantMin:      0.9, // log2(1)*0.1 = 0
			wantMax:      0.9,
		},
		{
			name:         "explicit with boost",
			level:        LevelExplicit,
			timesDerived: 7,   // log2(8)*0.1 = 0.3
			wantMin:      1.0, // 0.9 + 0.3 = 1.2, capped at 1.0
			wantMax:      1.0,
		},
		{
			name:         "contradiction no boost",
			level:        LevelContradiction,
			timesDerived: 0,
			wantMin:      0.3,
			wantMax:      0.3,
		},
		{
			name:         "score never exceeds 1.0",
			level:        LevelExplicit,
			timesDerived: 1000,
			wantMin:      1.0,
			wantMax:      1.0,
		},
		{
			name:         "boost capped at 0.3",
			level:        LevelInductive,
			timesDerived: 100,
			wantMin:      0.5 + 0.3,
			wantMax:      0.8,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obs := Observation{Level: tt.level, TimesDerived: tt.timesDerived}
			score := obs.Score()
			if score < tt.wantMin-math.SmallestNonzeroFloat64 || score > tt.wantMax+math.SmallestNonzeroFloat64 {
				t.Errorf("Score() = %v, want in [%v, %v]", score, tt.wantMin, tt.wantMax)
			}
		})
	}
}

func TestSearchModeZeroValue(t *testing.T) {
	// Verify SearchMode zero value is Hybrid (iota=0)
	// This ensures uninitialized SearchOptions default to the most
	// comprehensive search strategy.
	var mode SearchMode
	if mode != SearchModeHybrid {
		t.Errorf("zero SearchMode = %d, want %d (SearchModeHybrid)", mode, SearchModeHybrid)
	}
}
