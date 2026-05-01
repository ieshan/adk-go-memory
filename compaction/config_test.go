// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package compaction

import (
	"testing"

	"github.com/ieshan/adk-go-pkg/testutil"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

func TestConfig_Validate(t *testing.T) {
	tests := []struct {
		name    string
		config  Config
		wantErr bool
		errMsg  string
	}{
		{
			name: "valid config with MaxEvents",
			config: Config{
				Strategy:   &TruncationStrategy{RetainCount: 10},
				MaxEvents:  100,
				KeepRecent: 10,
			},
			wantErr: false,
		},
		{
			name: "valid config with MaxTokens",
			config: Config{
				Strategy:   &TruncationStrategy{RetainCount: 10},
				MaxTokens:  4000,
				KeepRecent: 10,
			},
			wantErr: false,
		},
		{
			name: "valid config with Interval",
			config: Config{
				Strategy:   &TruncationStrategy{RetainCount: 10},
				Interval:   50,
				KeepRecent: 10,
			},
			wantErr: false,
		},
		{
			name: "valid config with multiple triggers",
			config: Config{
				Strategy:   &TruncationStrategy{RetainCount: 10},
				MaxEvents:  100,
				MaxTokens:  4000,
				Interval:   50,
				KeepRecent: 10,
			},
			wantErr: false,
		},
		{
			name: "missing strategy",
			config: Config{
				MaxEvents:  100,
				KeepRecent: 10,
			},
			wantErr: true,
			errMsg:  "Strategy is required",
		},
		{
			name: "no triggers enabled",
			config: Config{
				Strategy:   &TruncationStrategy{RetainCount: 10},
				KeepRecent: 10,
			},
			wantErr: true,
			errMsg:  "at least one trigger",
		},
		{
			name: "negative MaxEvents",
			config: Config{
				Strategy:   &TruncationStrategy{RetainCount: 10},
				MaxEvents:  -1,
				KeepRecent: 10,
			},
			wantErr: true,
			errMsg:  "MaxEvents must be >= 0",
		},
		{
			name: "negative MaxTokens",
			config: Config{
				Strategy:   &TruncationStrategy{RetainCount: 10},
				MaxTokens:  -1,
				KeepRecent: 10,
			},
			wantErr: true,
			errMsg:  "MaxTokens must be >= 0",
		},
		{
			name: "negative Interval",
			config: Config{
				Strategy:   &TruncationStrategy{RetainCount: 10},
				Interval:   -1,
				KeepRecent: 10,
			},
			wantErr: true,
			errMsg:  "Interval must be >= 0",
		},
		{
			name: "negative KeepRecent",
			config: Config{
				Strategy:   &TruncationStrategy{RetainCount: 10},
				MaxEvents:  100,
				KeepRecent: -1,
			},
			wantErr: true,
			errMsg:  "KeepRecent must be >= 0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if err != nil && tt.errMsg != "" {
				if err.Error() == "" || len(err.Error()) < len(tt.errMsg) {
					t.Errorf("Validate() error message = %v, want to contain %v", err, tt.errMsg)
				}
			}
		})
	}
}

func TestConfig_StateKey(t *testing.T) {
	tests := []struct {
		name     string
		stateKey string
		wantKey  string
	}{
		{
			name:     "default key",
			stateKey: "",
			wantKey:  DefaultStateKey,
		},
		{
			name:     "custom key",
			stateKey: "my_custom_key",
			wantKey:  "my_custom_key",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{stateKey: tt.stateKey}
			got := cfg.StateKey()
			if got != tt.wantKey {
				t.Errorf("StateKey() = %v, want %v", got, tt.wantKey)
			}
		})
	}
}

func TestConfig_ShouldCompact_MaxEvents(t *testing.T) {
	cfg := &Config{
		MaxEvents:  10,
		KeepRecent: 2,
	}

	tests := []struct {
		name            string
		eventCount      int
		wantCompact     bool
		wantReasonMatch string
	}{
		{
			name:        "below threshold",
			eventCount:  5,
			wantCompact: false,
		},
		{
			name:        "at threshold",
			eventCount:  10,
			wantCompact: false,
		},
		{
			name:            "above threshold",
			eventCount:      11,
			wantCompact:     true,
			wantReasonMatch: "exceeds MaxEvents",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			events := make([]*session.Event, tt.eventCount)
			for i := 0; i < tt.eventCount; i++ {
				events[i] = &session.Event{}
			}
			sess := testutil.NewFakeSession().WithEvents(events...)
			compactionState := &state{}

			got, reason := cfg.ShouldCompact(sess, compactionState)
			if got != tt.wantCompact {
				t.Errorf("ShouldCompact() = %v, want %v", got, tt.wantCompact)
			}
			if tt.wantCompact && tt.wantReasonMatch != "" {
				if reason == "" || len(reason) < len(tt.wantReasonMatch) {
					t.Errorf("ShouldCompact() reason = %v, want to contain %v", reason, tt.wantReasonMatch)
				}
			}
		})
	}
}

func TestConfig_ShouldCompact_Interval(t *testing.T) {
	cfg := &Config{
		Interval:   10,
		KeepRecent: 2,
	}

	tests := []struct {
		name           string
		eventCount     int
		lastEventCount int
		wantCompact    bool
	}{
		{
			name:           "below interval",
			eventCount:     15,
			lastEventCount: 10,
			wantCompact:    false,
		},
		{
			name:           "at interval",
			eventCount:     20,
			lastEventCount: 10,
			wantCompact:    true,
		},
		{
			name:           "above interval",
			eventCount:     25,
			lastEventCount: 10,
			wantCompact:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			events := make([]*session.Event, tt.eventCount)
			for i := 0; i < tt.eventCount; i++ {
				events[i] = &session.Event{}
			}
			sess := testutil.NewFakeSession().WithEvents(events...)
			compactionState := &state{LastEventCount: tt.lastEventCount}

			got, _ := cfg.ShouldCompact(sess, compactionState)
			if got != tt.wantCompact {
				t.Errorf("ShouldCompact() = %v, want %v", got, tt.wantCompact)
			}
		})
	}
}

func TestConfig_ShouldCompact_MaxTokens(t *testing.T) {
	cfg := &Config{
		MaxTokens:  5, // Very low threshold for testing
		KeepRecent: 2,
	}

	tests := []struct {
		name            string
		events          []*session.Event
		wantCompact     bool
		wantReasonMatch string
	}{
		{
			name:        "below token threshold",
			events:      createEventsWithContent(2, "hi"), // ~1 token
			wantCompact: false,
		},
		{
			name:            "above token threshold",
			events:          createEventsWithContent(5, "hello world this is a longer message"), // ~7 tokens each
			wantCompact:     true,
			wantReasonMatch: "exceeds MaxTokens",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sess := testutil.NewFakeSession().WithEvents(tt.events...)
			compactionState := &state{}

			got, reason := cfg.ShouldCompact(sess, compactionState)
			if got != tt.wantCompact {
				t.Errorf("ShouldCompact() = %v, want %v", got, tt.wantCompact)
			}
			if tt.wantCompact && tt.wantReasonMatch != "" {
				if reason == "" || len(reason) < len(tt.wantReasonMatch) {
					t.Errorf("ShouldCompact() reason = %v, want to contain %v", reason, tt.wantReasonMatch)
				}
			}
		})
	}
}

func TestConfig_ShouldCompact_IntervalAfterCompaction(t *testing.T) {
	cfg := &Config{
		Interval:   10,
		KeepRecent: 2,
	}

	// Simulate: compaction happened at event count 20, now at 25
	// 25 - 20 = 5 new events, which is less than interval 10
	events := make([]*session.Event, 25)
	for i := 0; i < 25; i++ {
		events[i] = &session.Event{}
	}
	sess := testutil.NewFakeSession().WithEvents(events...)
	compactionState := &state{LastEventCount: 20}

	got, _ := cfg.ShouldCompact(sess, compactionState)
	if got {
		t.Error("ShouldCompact() = true, want false (only 5 new events since last compaction, interval is 10)")
	}

	// Now simulate: compaction happened at event count 20, now at 30
	// 30 - 20 = 10 new events, which equals interval 10
	events2 := make([]*session.Event, 30)
	for i := 0; i < 30; i++ {
		events2[i] = &session.Event{}
	}
	sess2 := testutil.NewFakeSession().WithEvents(events2...)
	got2, reason := cfg.ShouldCompact(sess2, compactionState)
	if !got2 {
		t.Error("ShouldCompact() = false, want true (10 new events since last compaction equals interval 10)")
	}
	if reason == "" {
		t.Error("Expected non-empty reason when compaction should occur")
	}
}

func TestConfig_ShouldCompact_NoTriggers(t *testing.T) {
	cfg := &Config{
		KeepRecent: 2,
		// No MaxEvents, MaxTokens, or Interval set
	}

	events := make([]*session.Event, 100)
	for i := 0; i < 100; i++ {
		events[i] = &session.Event{}
	}
	sess := testutil.NewFakeSession().WithEvents(events...)
	compactionState := &state{}

	got, _ := cfg.ShouldCompact(sess, compactionState)
	if got {
		t.Error("ShouldCompact() = true, want false (no triggers configured)")
	}
}

func TestConfig_estimateTokens(t *testing.T) {
	cfg := &Config{}

	tests := []struct {
		name       string
		events     []*session.Event
		startIndex int
		wantApprox int // approximate range due to floating point
	}{
		{
			name:       "empty events",
			events:     []*session.Event{},
			startIndex: 0,
			wantApprox: 0,
		},
		{
			name:       "single event",
			events:     createEventsWithContent(1, "hello"), // 5 chars
			startIndex: 0,
			wantApprox: 1, // ~1 token
		},
		{
			name:       "multiple events",
			events:     createEventsWithContent(2, "hello world test"), // 16 chars each
			startIndex: 0,
			wantApprox: 8, // ~8 tokens (32 chars * 0.25)
		},
		{
			name:       "with start index offset",
			events:     createEventsWithContent(3, "hello"),
			startIndex: 1,
			wantApprox: 1, // only 2 events counted (at index 1 and 2)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sess := testutil.NewFakeSession().WithEvents(tt.events...)
			got := cfg.estimateTokens(sess, tt.startIndex)
			// Allow some tolerance for token estimation
			diff := got - tt.wantApprox
			if diff < -2 || diff > 2 {
				t.Errorf("estimateTokens() = %v, want approximately %v", got, tt.wantApprox)
			}
		})
	}
}

func TestGetCompactionState(t *testing.T) {
	tests := []struct {
		name       string
		key        string
		stateValue interface{}
		wantErr    bool
	}{
		{
			name:       "empty key uses default",
			key:        "",
			stateValue: nil,
			wantErr:    false,
		},
		{
			name:       "valid state",
			key:        "test_key",
			stateValue: &state{LastCompactedIndex: 5},
			wantErr:    false,
		},
		{
			name:       "invalid type",
			key:        "test_key",
			stateValue: "invalid string type",
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state := testutil.NewFakeStateWithData(map[string]any{tt.key: tt.stateValue})
			sess := testutil.NewFakeSession().WithState(state.Data)
			_, err := GetCompactionState(sess, tt.key)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetCompactionState() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestSaveCompactionState(t *testing.T) {
	sess := testutil.NewFakeSession()
	st := &state{
		LastCompactedIndex: 10,
		TotalCompactions:   5,
	}

	err := SaveCompactionState(sess, "test_key", st)
	if err != nil {
		t.Errorf("SaveCompactionState() error = %v", err)
	}
}

func TestGetEventsForCompaction(t *testing.T) {
	tests := []struct {
		name             string
		eventCount       int
		lastCompactedIdx int
		keepRecent       int
		wantLen          int
	}{
		{
			name:             "no events",
			eventCount:       0,
			lastCompactedIdx: 0,
			keepRecent:       2,
			wantLen:          0,
		},
		{
			name:             "all events compacted",
			eventCount:       5,
			lastCompactedIdx: 4,
			keepRecent:       2,
			wantLen:          0,
		},
		{
			name:             "not enough to keep recent",
			eventCount:       5,
			lastCompactedIdx: 2,
			keepRecent:       5,
			wantLen:          0,
		},
		{
			name:             "normal case",
			eventCount:       10,
			lastCompactedIdx: 3,
			keepRecent:       2,
			wantLen:          4, // events 4,5,6,7 (indices 8,9 are kept)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			events := make([]*session.Event, tt.eventCount)
			for i := 0; i < tt.eventCount; i++ {
				events[i] = &session.Event{
					ID:     string(rune('a' + i)),
					Author: "user",
				}
				events[i].Content = genai.NewContentFromText("test", genai.RoleUser)
			}
			sess := testutil.NewFakeSession().WithEvents(events...)

			got := GetEventsForCompaction(sess, tt.lastCompactedIdx, tt.keepRecent)
			if len(got) != tt.wantLen {
				t.Errorf("GetEventsForCompaction() returned %d events, want %d", len(got), tt.wantLen)
			}
		})
	}
}

func TestNewConfig(t *testing.T) {
	strategy := &TruncationStrategy{RetainCount: 10}

	tests := []struct {
		name    string
		opts    []ConfigOption
		wantErr bool
	}{
		{
			name: "valid config with option",
			opts: []ConfigOption{
				WithMaxEvents(100),
				WithMaxTokens(4000),
				WithInterval(50),
				WithKeepRecent(20),
				WithEnableAfterAgent(true),
				WithStateKey("custom_key"),
			},
			wantErr: false,
		},
		{
			name:    "no options fails validation (no triggers set)",
			opts:    []ConfigOption{},
			wantErr: true, // fails validation because no triggers
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := NewConfig(strategy, tt.opts...)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewConfig() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if err == nil {
				if cfg.Strategy != strategy {
					t.Error("NewConfig() Strategy not set correctly")
				}
				if cfg.KeepRecent != 20 && len(tt.opts) > 0 {
					t.Errorf("NewConfig() KeepRecent = %v, want 20", cfg.KeepRecent)
				}
			}
		})
	}
}

func TestSummaryEvent(t *testing.T) {
	summary := "Test summary content"
	event := SummaryEvent(summary)

	if event.Author != "system" {
		t.Errorf("SummaryEvent() Author = %v, want system", event.Author)
	}
	if event.Timestamp.IsZero() {
		t.Error("SummaryEvent() Timestamp should be set")
	}
	if event.Content == nil {
		t.Fatal("SummaryEvent() Content should not be nil")
	}
	if len(event.Content.Parts) == 0 {
		t.Fatal("SummaryEvent() Content.Parts should not be empty")
	}
	if event.Content.Parts[0].Text != "[Previous conversation summarized]: "+summary {
		t.Errorf("SummaryEvent() text = %v, want prefix + summary", event.Content.Parts[0].Text)
	}
}

// Helper functions for testing

func createEventsWithContent(n int, content string) []*session.Event {
	events := make([]*session.Event, n)
	for i := 0; i < n; i++ {
		events[i] = &session.Event{
			ID:     string(rune('a' + i)),
			Author: "user",
		}
		events[i].Content = genai.NewContentFromText(content, genai.RoleUser)
	}
	return events
}
