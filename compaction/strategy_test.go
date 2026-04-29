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
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/ieshan/adk-go-memory/internal/testutil"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

func TestTruncationStrategy_Compact(t *testing.T) {
	tests := []struct {
		name        string
		retainCount int
		events      []*session.Event
		wantLen     int
		wantErr     bool
	}{
		{
			name:        "truncate to last 3",
			retainCount: 3,
			events:      createEvents(5),
			wantLen:     3,
			wantErr:     false,
		},
		{
			name:        "no truncation needed",
			retainCount: 10,
			events:      createEvents(5),
			wantLen:     5,
			wantErr:     false,
		},
		{
			name:        "empty events",
			retainCount: 5,
			events:      []*session.Event{},
			wantLen:     0,
			wantErr:     false,
		},
		{
			name:        "invalid retain count",
			retainCount: 0,
			events:      createEvents(5),
			wantLen:     0,
			wantErr:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			strategy := &TruncationStrategy{RetainCount: tt.retainCount}
			got, err := strategy.Compact(context.Background(), tt.events)
			if (err != nil) != tt.wantErr {
				t.Errorf("Compact() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if len(got) != tt.wantLen {
				t.Errorf("Compact() got %d events, want %d", len(got), tt.wantLen)
			}
		})
	}
}

func TestIdentityStrategy_Compact(t *testing.T) {
	events := createEvents(5)
	strategy := &IdentityStrategy{}

	got, err := strategy.Compact(context.Background(), events)
	if err != nil {
		t.Errorf("IdentityStrategy.Compact() error = %v", err)
	}
	if len(got) != len(events) {
		t.Errorf("IdentityStrategy.Compact() returned %d events, want %d", len(got), len(events))
	}
}

func TestCompositeStrategy_Compact(t *testing.T) {
	events := createEvents(10)

	// First truncate to 5, then identity (should keep 5)
	strategy := &CompositeStrategy{
		Strategies: []Strategy{
			&TruncationStrategy{RetainCount: 5},
			&IdentityStrategy{},
		},
	}

	got, err := strategy.Compact(context.Background(), events)
	if err != nil {
		t.Errorf("CompositeStrategy.Compact() error = %v", err)
	}
	if len(got) != 5 {
		t.Errorf("CompositeStrategy.Compact() returned %d events, want 5", len(got))
	}
}

func TestCompositeStrategy_ErrorPropagation(t *testing.T) {
	strategy := &CompositeStrategy{
		Strategies: []Strategy{
			&TruncationStrategy{RetainCount: 0}, // Will error
		},
	}

	_, err := strategy.Compact(context.Background(), createEvents(5))
	if err == nil {
		t.Error("CompositeStrategy.Compact() expected error, got nil")
	}
}

func TestSummarizationStrategy_MissingLLM(t *testing.T) {
	strategy := &SummarizationStrategy{LLM: nil}
	_, err := strategy.Compact(context.Background(), createEvents(3))
	if err == nil {
		t.Error("SummarizationStrategy.Compact() expected error for nil LLM, got nil")
	}
}

func TestSummarizationStrategy_EmptyEvents(t *testing.T) {
	strategy := &SummarizationStrategy{LLM: &testutil.FakeGenaiLLM{}}
	got, err := strategy.Compact(context.Background(), []*session.Event{})
	if err != nil {
		t.Errorf("SummarizationStrategy.Compact() error = %v", err)
	}
	if len(got) != 0 {
		t.Errorf("SummarizationStrategy.Compact() returned %d events, want 0", len(got))
	}
}

func TestSummarizationStrategy_Compact_WithLLM(t *testing.T) {
	llm := &testutil.FakeGenaiLLM{Response: "Key facts: user likes Go, prefers dark mode"}
	strategy := &SummarizationStrategy{LLM: llm}

	events := createEvents(5)
	got, err := strategy.Compact(context.Background(), events)
	if err != nil {
		t.Fatalf("SummarizationStrategy.Compact() error = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("SummarizationStrategy.Compact() returned %d events, want 1", len(got))
	}
	if got[0].Author != "system" {
		t.Errorf("Summary event Author = %q, want 'system'", got[0].Author)
	}
	if got[0].Content == nil || len(got[0].Content.Parts) == 0 {
		t.Fatal("Summary event should have content")
	}
	if got[0].Content.Parts[0].Text != "Key facts: user likes Go, prefers dark mode" {
		t.Errorf("Summary text = %q, want LLM response text", got[0].Content.Parts[0].Text)
	}
}

func TestSummarizationStrategy_Compact_CustomInstruction(t *testing.T) {
	llm := &testutil.FakeGenaiLLM{Response: "Custom summary"}
	strategy := &SummarizationStrategy{
		LLM:         llm,
		Instruction: "Custom prompt: %s",
	}

	events := createEvents(3)
	got, err := strategy.Compact(context.Background(), events)
	if err != nil {
		t.Fatalf("SummarizationStrategy.Compact() error = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("Expected 1 event, got %d", len(got))
	}
}

func TestSummarizationStrategy_Compact_ContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	strategy := &SummarizationStrategy{LLM: &testutil.FakeGenaiLLM{}}
	_, err := strategy.Compact(ctx, createEvents(3))
	if err == nil {
		t.Error("Expected error for cancelled context, got nil")
	}
}

func TestSummarizationStrategy_Compact_EmptySummaryResponse(t *testing.T) {
	llm := &testutil.FakeGenaiLLM{Response: "", AllowEmpty: true}
	strategy := &SummarizationStrategy{LLM: llm}

	_, err := strategy.Compact(context.Background(), createEvents(3))
	if err == nil {
		t.Error("Expected error for empty LLM summary, got nil")
	}
}

func TestSummarizationStrategy_Compact_LLMError(t *testing.T) {
	strategy := &SummarizationStrategy{
		LLM: &testutil.ErrorGenaiLLM{},
	}

	_, err := strategy.Compact(context.Background(), createEvents(3))
	if err == nil {
		t.Error("Expected error for LLM failure, got nil")
	}
}

// Helper functions

func createEvents(n int) []*session.Event {
	events := make([]*session.Event, n)
	for i := 0; i < n; i++ {
		events[i] = &session.Event{
			ID:        fmt.Sprintf("event-%d", i),
			Author:    "user",
			Timestamp: time.Now().Add(time.Duration(i) * time.Second),
		}
		events[i].Content = genai.NewContentFromText(fmt.Sprintf("Message %d", i), genai.RoleUser)
	}
	return events
}
