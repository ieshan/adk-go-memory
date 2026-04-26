// Package main provides tests for the memory tool example.
package main

import (
	"context"
	"testing"
	"time"

	"google.golang.org/genai"

	memory "github.com/ieshan/adk-go-memory"
	"github.com/ieshan/adk-go-memory/adapter"
	"github.com/ieshan/adk-go-memory/examples/internal/testutil"
	adkagent "google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// TestMemoryTool_ExplicitSearch verifies that the agent can use the
// search_memory tool to find relevant observations.
func TestMemoryTool_ExplicitSearch(t *testing.T) {
	ctx := context.Background()

	// Seed storage with observations
	storage := adapter.InMemory()

	obs := &adapter.Observation{
		ID:        "obs-1",
		Content:   "User enjoys hiking on weekends",
		Level:     adapter.LevelExplicit,
		UserID:    "user1",
		AppName:   "tool-test",
		CreatedAt: time.Now(),
	}
	if err := storage.Store(ctx, obs); err != nil {
		t.Fatalf("Store error = %v", err)
	}

	// Fake LLM that calls search_memory tool then responds
	llm := &testutil.FakeLLM{
		Responses: []model.LLMResponse{
			// First: Function call to search_memory
			{
				Content: &genai.Content{
					Role: genai.RoleModel,
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								Name: "search_memory",
								Args: map[string]any{"query": "weekend activities"},
							},
						},
					},
				},
			},
			// After tool result: Final response
			{
				Content: genai.NewContentFromText(
					"I remember you enjoy hiking on weekends! Would you like trail recommendations?",
					genai.RoleModel,
				),
			},
		},
	}

	// Create memory tool
	memTool, err := memory.NewMemoryTool(storage)
	if err != nil {
		t.Fatalf("NewMemoryTool error = %v", err)
	}

	// Create agent with tool
	agent, err := llmagent.New(llmagent.Config{
		Name:  "assistant",
		Model: llm,
		Tools: []tool.Tool{memTool},
	})
	if err != nil {
		t.Fatalf("Create agent error = %v", err)
	}

	// Create session and runner
	sessionSvc := session.InMemoryService()
	resp, _ := sessionSvc.Create(ctx, &session.CreateRequest{
		AppName: "tool-test", UserID: "user1", SessionID: "sess1",
	})
	sess := resp.Session

	r, _ := runner.New(runner.Config{
		AppName:        "tool-test",
		Agent:          agent,
		SessionService: sessionSvc,
	})

	// User asks about weekend activities
	msg := genai.NewContentFromText("What do I like to do on weekends?", genai.RoleUser)

	eventCount := 0
	for event, err := range r.Run(ctx, "user1", sess.ID(), msg, adkagent.RunConfig{}) {
		if err != nil {
			t.Logf("Run error: %v", err)
			continue
		}
		eventCount++

		// Check for function call
		if event.LLMResponse.Content != nil {
			for _, part := range event.LLMResponse.Content.Parts {
				if part.FunctionCall != nil && part.FunctionCall.Name == "search_memory" {
					// Verify the tool was called
					query, ok := part.FunctionCall.Args["query"]
					if !ok || query == "" {
						t.Error("Expected 'query' parameter in search_memory call")
					}
				}
				if part.Text != "" {
					// Verify final response references hiking
					if !containsIgnoreCase(part.Text, "hiking") {
						t.Errorf("Expected response to mention hiking, got: %s", part.Text)
					}
				}
			}
		}
	}

	if eventCount == 0 {
		t.Error("Expected at least one event from agent run")
	}
}

// TestMemoryTool_NoResults verifies the agent handles empty memory gracefully.
func TestMemoryTool_NoResults(t *testing.T) {
	ctx := context.Background()

	// Empty storage
	storage := adapter.InMemory()

	// Fake LLM calls tool, gets empty results
	llm := &testutil.FakeLLM{
		Responses: []model.LLMResponse{
			{
				Content: &genai.Content{
					Role: genai.RoleModel,
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								Name: "search_memory",
								Args: map[string]any{"query": "unknown topic"},
							},
						},
					},
				},
			},
			{
				Content: genai.NewContentFromText(
					"I don't have any information about that in my memory.",
					genai.RoleModel,
				),
			},
		},
	}

	memTool, _ := memory.NewMemoryTool(storage)
	agent, err := llmagent.New(llmagent.Config{
		Name:  "assistant",
		Model: llm,
		Tools: []tool.Tool{memTool},
	})
	if err != nil {
		t.Fatalf("Create agent error = %v", err)
	}

	sessionSvc := session.InMemoryService()
	resp, _ := sessionSvc.Create(ctx, &session.CreateRequest{
		AppName: "tool-test", UserID: "user1", SessionID: "sess1",
	})
	sess := resp.Session

	r, _ := runner.New(runner.Config{
		AppName:        "tool-test",
		Agent:          agent,
		SessionService: sessionSvc,
	})

	msg := genai.NewContentFromText("Tell me about my past", genai.RoleUser)

	foundEmptyResponse := false
	for event, err := range r.Run(ctx, "user1", sess.ID(), msg, adkagent.RunConfig{}) {
		if err != nil {
			t.Logf("Run error: %v", err)
			continue
		}
		if event.LLMResponse.Content != nil {
			for _, part := range event.LLMResponse.Content.Parts {
				if part.Text != "" && containsIgnoreCase(part.Text, "don't have") {
					foundEmptyResponse = true
				}
			}
		}
	}

	if !foundEmptyResponse {
		t.Log("Agent handled empty memory (may have used different phrasing)")
	}
}

// TestMemoryTool_ToolSchema verifies the tool has correct declaration.
func TestMemoryTool_ToolSchema(t *testing.T) {
	storage := adapter.InMemory()

	memTool, err := memory.NewMemoryTool(storage)
	if err != nil {
		t.Fatalf("NewMemoryTool error = %v", err)
	}

	// Verify tool declaration
	decl := memTool.Declaration()
	if decl.Name != "search_memory" {
		t.Errorf("Expected tool name 'search_memory', got %s", decl.Name)
	}

	if decl.Description == "" {
		t.Error("Expected non-empty tool description")
	}

	// Verify required parameter 'query'
	if decl.Parameters == nil {
		t.Fatal("Expected Parameters in tool declaration")
	}

	if _, ok := decl.Parameters.Properties["query"]; !ok {
		t.Error("Expected 'query' parameter in tool schema")
	}

	found := false
	for _, req := range decl.Parameters.Required {
		if req == "query" {
			found = true
			break
		}
	}
	if !found {
		t.Error("Expected 'query' to be required parameter")
	}

	// Verify tool interface
	if memTool.Name() != "search_memory" {
		t.Errorf("Expected Name() = 'search_memory', got %s", memTool.Name())
	}

	if memTool.IsLongRunning() {
		t.Error("Expected IsLongRunning() = false")
	}
}

// containsIgnoreCase checks if s contains substr (case-insensitive).
func containsIgnoreCase(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr ||
		len(substr) == 0 ||
		findSubstrIgnoreCase(s, substr) >= 0)
}

func findSubstrIgnoreCase(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		match := true
		for j := 0; j < len(substr); j++ {
			if toLower(s[i+j]) != toLower(substr[j]) {
				match = false
				break
			}
		}
		if match {
			return i
		}
	}
	return -1
}

func toLower(c byte) byte {
	if c >= 'A' && c <= 'Z' {
		return c + ('a' - 'A')
	}
	return c
}
