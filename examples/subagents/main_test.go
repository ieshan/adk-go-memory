// Package main provides tests for the sub-agents example.
package main

import (
	"context"
	"testing"
	"time"

	"google.golang.org/genai"

	memory "github.com/ieshan/adk-go-memory"
	"github.com/ieshan/adk-go-memory/adapter"
	"github.com/ieshan/adk-go-memory/examples/internal/testutil"
	"google.golang.org/adk/model"
)

// TestSubAgents_MemoryExtraction verifies that observations are extracted
// from conversation events across multiple agents.
func TestSubAgents_MemoryExtraction(t *testing.T) {
	ctx := context.Background()

	// Fake LLM for deriver to extract observations
	llm := &testutil.FakeLLM{
		Responses: []model.LLMResponse{
			{
				Content: genai.NewContentFromText(
					`{"observations":[{"content":"user name is Bob","level":"explicit"}]}`,
					genai.RoleModel,
				),
			},
		},
	}

	// Setup storage + memory service
	storage, err := adapter.InMemory()
	if err != nil {
		t.Fatalf("InMemory() error = %v", err)
	}
	defer storage.Close()

	deriver := memory.NewDeriver(memory.DeriverConfig{LLM: llm, Storage: storage})

	// Directly test deriver with messages from multiple agents
	msgs := []memory.TimestampedMessage{
		{Content: genai.NewContentFromText("My name is Bob.", genai.RoleUser), At: time.Now()},
		{Content: genai.NewContentFromText("I'll remember your name is Bob!", genai.RoleModel), At: time.Now()},
	}

	if err := deriver.Derive(ctx, msgs, "subagent-session", "", ""); err != nil {
		t.Fatalf("Derive error = %v", err)
	}

	// Verify observation was extracted
	results, _ := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "Bob",
		Mode:       adapter.SearchModeFTS,
		MaxResults: 10,
	})
	if len(results) == 0 {
		t.Error("Expected observation about Bob to be extracted")
	}
}

// TestSubAgents_MemoryExtractedFromAllAgents verifies that observations
// are extracted from messages authored by any agent in the tree.
func TestSubAgents_MemoryExtractedFromAllAgents(t *testing.T) {
	ctx := context.Background()

	// Fake LLM returns observations for all messages in a single batched call
	llm := &testutil.FakeLLM{
		Responses: []model.LLMResponse{
			{
				Content: genai.NewContentFromText(
					`{"observations":[{"content":"user lives in NYC","level":"explicit"},{"content":"user prefers morning greetings","level":"deductive"}]}`,
					genai.RoleModel,
				),
			},
		},
	}

	storage, _ := adapter.InMemory()
	defer storage.Close()

	deriver := memory.NewDeriver(memory.DeriverConfig{LLM: llm, Storage: storage})

	// Simulate messages from multiple agents (fact_recorder and greeter)
	msgs := []memory.TimestampedMessage{
		{Content: genai.NewContentFromText("I see you live in NYC.", genai.RoleModel), At: time.Now()},
		{Content: genai.NewContentFromText("Good morning! You're an early bird.", genai.RoleModel), At: time.Now()},
	}

	if err := deriver.Derive(ctx, msgs, "multi-agent-session", "", ""); err != nil {
		t.Fatalf("Derive error = %v", err)
	}

	// Verify both observations extracted
	results, _ := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "NYC",
		Mode:       adapter.SearchModeFTS,
		MaxResults: 10,
	})
	if len(results) == 0 {
		t.Error("Expected observation about NYC")
	}

	results, _ = storage.Search(ctx, &adapter.SearchOptions{
		Query:      "morning",
		Mode:       adapter.SearchModeFTS,
		MaxResults: 10,
	})
	if len(results) == 0 {
		t.Error("Expected observation about morning greetings")
	}
}
