// Package main provides tests for the HTTP tool example.
package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"google.golang.org/genai"

	"github.com/ieshan/adk-go-memory/examples/internal/testutil"
	adkagent "google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// TestHTTPTool_MockServer verifies that the weather tool works correctly
// with a mocked HTTP server.
func TestHTTPTool_MockServer(t *testing.T) {
	// Create mock weather server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		city := r.URL.Query().Get("city")
		if city == "" {
			http.Error(w, "city required", http.StatusBadRequest)
			return
		}

		// Return mock weather data
		response := WeatherResult{
			City:        city,
			Temperature: 72,
			Condition:   "sunny",
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}))
	defer server.Close()

	// Create weather tool that calls the mock server
	weatherTool := NewWeatherTool(http.DefaultClient, server.URL)

	// Test the tool directly
	// In a real scenario, tool.Context would be passed by ADK framework
	result, err := weatherTool.Run(nil, map[string]any{"city": "San Francisco"})
	if err != nil {
		t.Fatalf("Weather tool error: %v", err)
	}

	// Verify the result
	if result["city"] != "San Francisco" {
		t.Errorf("Expected city 'San Francisco', got %v", result["city"])
	}
	if result["temperature"] != 72.0 {
		t.Errorf("Expected temperature 72, got %v", result["temperature"])
	}
	if result["condition"] != "sunny" {
		t.Errorf("Expected condition 'sunny', got %v", result["condition"])
	}
}

// TestHTTPTool_WithAgent verifies the full agent flow with the HTTP tool.
func TestHTTPTool_WithAgent(t *testing.T) {
	// Create mock weather server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		city := r.URL.Query().Get("city")
		response := WeatherResult{
			City:        city,
			Temperature: 65,
			Condition:   "partly cloudy",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	// Create weather tool
	weatherTool := NewWeatherTool(http.DefaultClient, server.URL)

	// Fake LLM calls the weather tool
	llm := &testutil.FakeLLM{
		Responses: []model.LLMResponse{
			// First: Function call to get_weather
			{
				Content: &genai.Content{
					Role: genai.RoleModel,
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								Name: "get_weather",
								Args: map[string]any{"city": "New York"},
							},
						},
					},
				},
			},
			// After tool result: Final response
			{
				Content: genai.NewContentFromText(
					"It's 65°F and partly cloudy in New York!",
					genai.RoleModel,
				),
			},
		},
	}

	// Create agent with weather tool
	agent, err := llmagent.New(llmagent.Config{
		Name:        "weather_assistant",
		Model:       llm,
		Description: "Assistant with weather lookup",
		Instruction: "Use the get_weather tool to check weather.",
		Tools:       []tool.Tool{weatherTool},
	})
	if err != nil {
		t.Fatalf("Create agent error = %v", err)
	}

	// Create session and runner
	ctx := context.Background()
	sessionSvc := session.InMemoryService()
	resp, _ := sessionSvc.Create(ctx, &session.CreateRequest{
		AppName: "weather-test", UserID: "user1", SessionID: "sess1",
	})
	sess := resp.Session

	r, _ := runner.New(runner.Config{
		AppName:        "weather-test",
		Agent:          agent,
		SessionService: sessionSvc,
	})

	// User asks about weather
	msg := genai.NewContentFromText("What's the weather in New York?", genai.RoleUser)

	foundToolCall := false
	foundFinalResponse := false

	for event, err := range r.Run(ctx, "user1", sess.ID(), msg, adkagent.RunConfig{}) {
		if err != nil {
			t.Logf("Run error: %v", err)
			continue
		}

		if event.LLMResponse.Content != nil {
			for _, part := range event.LLMResponse.Content.Parts {
				// Check for function call
				if part.FunctionCall != nil {
					if part.FunctionCall.Name == "get_weather" {
						foundToolCall = true
						city, ok := part.FunctionCall.Args["city"]
						if !ok || city != "New York" {
							t.Errorf("Expected city 'New York', got %v", city)
						}
					}
				}

				// Check for final response
				if part.Text != "" && containsText(part.Text, "New York") {
					foundFinalResponse = true
				}
			}
		}
	}

	if !foundToolCall {
		t.Error("Expected agent to call get_weather tool")
	}
	if !foundFinalResponse {
		t.Error("Expected agent to include weather in final response")
	}
}

// TestHTTPTool_ToolSchema verifies the weather tool schema.
func TestHTTPTool_ToolSchema(t *testing.T) {
	tool := NewWeatherTool(nil, "https://example.com")

	// Verify basic properties
	if tool.Name() != "get_weather" {
		t.Errorf("Expected name 'get_weather', got %s", tool.Name())
	}

	if tool.IsLongRunning() {
		t.Error("Expected IsLongRunning = false")
	}

	// Verify declaration
	decl := tool.Declaration()
	if decl.Name != "get_weather" {
		t.Errorf("Expected declaration name 'get_weather', got %s", decl.Name)
	}

	if decl.Parameters == nil {
		t.Fatal("Expected Parameters in declaration")
	}

	// Verify 'city' parameter exists and is required
	if _, ok := decl.Parameters.Properties["city"]; !ok {
		t.Error("Expected 'city' parameter in schema")
	}

	found := false
	for _, req := range decl.Parameters.Required {
		if req == "city" {
			found = true
			break
		}
	}
	if !found {
		t.Error("Expected 'city' to be required")
	}
}

// TestHTTPTool_InvalidCity verifies error handling for invalid input.
func TestHTTPTool_InvalidCity(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer server.Close()

	tool := NewWeatherTool(http.DefaultClient, server.URL)

	// Test with empty city
	_, err := tool.Run(nil, map[string]any{})
	if err == nil {
		t.Error("Expected error for missing city")
	}

	// Test with nil city
	_, err = tool.Run(nil, map[string]any{"city": nil})
	if err == nil {
		t.Error("Expected error for nil city")
	}
}

// containsText checks if s contains substr (case-insensitive).
func containsText(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || findSubstr(s, substr) >= 0)
}

func findSubstr(s, substr string) int {
	sLower := toLowerString(s)
	substrLower := toLowerString(substr)

	for i := 0; i <= len(sLower)-len(substrLower); i++ {
		if sLower[i:i+len(substrLower)] == substrLower {
			return i
		}
	}
	return -1
}

func toLowerString(s string) string {
	result := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c = c + ('a' - 'A')
		}
		result[i] = c
	}
	return string(result)
}
