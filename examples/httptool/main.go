// Package main demonstrates an agent with an external HTTP tool.
//
// This example shows how to:
//   - Create a custom tool that makes HTTP requests
//   - Mock HTTP calls using httptest for testing
//   - Wire the tool into an agent with memory
//
// The weather tool makes HTTP requests to a weather API. In production,
// it would call a real API. In tests, it's mocked using httptest.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"

	"google.golang.org/genai"

	memory "github.com/ieshan/adk-go-memory"
	"github.com/ieshan/adk-go-memory/adapter"
	"github.com/ieshan/adk-go-memory/adapter/sqlite"
	adkagent "google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	adkmodel "google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

// WeatherTool provides weather information for a city.
type WeatherTool struct {
	httpClient *http.Client
	baseURL    string
}

// WeatherResult is the output of the weather tool.
type WeatherResult struct {
	City        string  `json:"city"`
	Temperature float64 `json:"temperature"`
	Condition   string  `json:"condition"`
}

// NewWeatherTool creates a new weather tool.
// In production, baseURL would be a real API endpoint.
// In tests, baseURL points to an httptest.Server.
func NewWeatherTool(httpClient *http.Client, baseURL string) *WeatherTool {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &WeatherTool{
		httpClient: httpClient,
		baseURL:    baseURL,
	}
}

// Name implements tool.Tool.
func (w *WeatherTool) Name() string {
	return "get_weather"
}

// Description implements tool.Tool.
func (w *WeatherTool) Description() string {
	return "Gets current weather for a city. Usage: get_weather(city: string)"
}

// IsLongRunning implements tool.Tool.
func (w *WeatherTool) IsLongRunning() bool {
	return false
}

// Declaration returns the GenAI FunctionDeclaration for the tool.
func (w *WeatherTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        w.Name(),
		Description: w.Description(),
		Parameters: &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				"city": {
					Type:        "STRING",
					Description: "City name to get weather for",
				},
			},
			Required: []string{"city"},
		},
	}
}

// Run executes the weather lookup.
func (w *WeatherTool) Run(ctx tool.Context, args map[string]any) (map[string]any, error) {
	city, ok := args["city"].(string)
	if !ok || city == "" {
		return nil, fmt.Errorf("city parameter is required")
	}

	// Make HTTP request with URL-encoded city
	reqURL := fmt.Sprintf("%s/weather?city=%s", w.baseURL, url.QueryEscape(city))
	resp, err := w.httpClient.Get(reqURL)
	if err != nil {
		return nil, fmt.Errorf("weather API request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("weather API returned status %d", resp.StatusCode)
	}

	// Parse response
	var result WeatherResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode weather response: %w", err)
	}

	return map[string]any{
		"city":        result.City,
		"temperature": result.Temperature,
		"condition":   result.Condition,
	}, nil
}

// ProcessRequest implements the tool request processor interface.
// It packs the tool declaration into the LLM request.
func (w *WeatherTool) ProcessRequest(ctx tool.Context, req *adkmodel.LLMRequest) error {
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	decl := w.Declaration()
	req.Config.Tools = append(req.Config.Tools, &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{decl},
	})
	return nil
}

func main() {
	ctx := context.Background()

	// Setup storage and memory kit
	storage, err := sqlite.InMemory()
	if err != nil {
		log.Fatalf("Failed to create SQLite storage: %v", err)
	}

	modelLLM := getLLM()

	// Create memory kit with all components
	kit, err := memory.New(memory.KitConfig{
		Storage: storage,
		LLM:     modelLLM,
	})
	if err != nil {
		log.Fatalf("Failed to create memory kit: %v", err)
	}
	defer kit.Close()

	// Create weather tool with default HTTP client
	// In production, this would call a real weather API
	weatherTool := NewWeatherTool(http.DefaultClient, "https://api.weather.example.com")

	// Create agent with weather tool
	agent, err := llmagent.New(llmagent.Config{
		Name:        "weather_assistant",
		Model:       modelLLM,
		Description: "Assistant that can look up weather information",
		Instruction: `You are a helpful assistant with access to weather information.

When users ask about the weather, use the get_weather tool to look it up.
The tool takes a city name as input and returns temperature and conditions.`,
		Tools: []tool.Tool{weatherTool},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// Create runner
	appName := "weather_app"
	sessionService := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:           appName,
		Agent:             agent,
		SessionService:    sessionService,
		MemoryService:     kit.Service,
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	// Interactive session
	userID := "user1"
	reader := bufio.NewReader(os.Stdin)

	fmt.Println("Weather Tool Demo")
	fmt.Println("I can look up weather using an HTTP tool (mocked in tests).")
	fmt.Println("Type 'quit' to exit, 'remember' to see stored memories.")
	fmt.Println("Try: 'What's the weather in San Francisco?'")
	fmt.Println()

	for {
		fmt.Print("You: ")
		userInput, err := reader.ReadString('\n')
		if err != nil {
			log.Printf("Error reading input: %v", err)
			continue
		}
		userInput = strings.TrimSpace(userInput)

		if userInput == "quit" {
			fmt.Println("Goodbye!")
			break
		}

		if userInput == "remember" {
			showMemory(ctx, storage, userID, appName)
			continue
		}

		msg := genai.NewContentFromText(userInput, genai.RoleUser)
		fmt.Print("Assistant: ")

		for event, err := range r.Run(ctx, userID, "", msg, adkagent.RunConfig{}) {
			if err != nil {
				fmt.Printf("\nError: %v\n", err)
				continue
			}
			if event.LLMResponse.Content != nil {
				for _, part := range event.LLMResponse.Content.Parts {
					if part.Text != "" {
						fmt.Print(part.Text)
					}
					if part.FunctionCall != nil {
						fmt.Printf("\n[Calling tool: %s]\n", part.FunctionCall.Name)
					}
				}
			}
		}
		fmt.Println()
	}
}

// getLLM returns the LLM to use.
func getLLM() model.LLM {
	return nil
}

// showMemory displays what the system remembers.
func showMemory(ctx context.Context, storage adapter.Storage, userID, appName string) {
	results, err := storage.Search(ctx, &adapter.SearchOptions{
		Query:      "",
		MaxResults: 10,
		Mode:       adapter.SearchModeFTS,
		UserID:     userID,
		AppName:    appName,
	})
	if err != nil {
		fmt.Printf("Error searching memory: %v\n", err)
		return
	}

	if len(results) == 0 {
		fmt.Println("No memories stored yet.")
		return
	}

	fmt.Println("Stored memories:")
	for _, r := range results {
		fmt.Printf("  - %s\n", r.Observation.Content)
	}
}
