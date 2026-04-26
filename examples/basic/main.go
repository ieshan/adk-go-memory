// Package main demonstrates a basic agent with memory integration using ADK-Go v1.2.0 patterns.
//
// This example shows how to:
//   - Set up storage for observations (in-memory map or SQLite adapter)
//   - Create a Deriver for automatic fact extraction from conversations
//   - Wire the memory service into an ADK-Go agent using MemoryKit
//   - Use memory tools (preload_memory and search_memory) for automatic and explicit memory access
//
// The agent automatically extracts observations from conversations and can
// recall them in subsequent interactions.
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/genai"

	memory "github.com/ieshan/adk-go-memory"
	"github.com/ieshan/adk-go-memory/adapter"
	"github.com/ieshan/adk-go-memory/adapter/sqlite"
	adkagent "google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

func main() {
	ctx := context.Background()

	// Setup in-memory SQLite-backed storage
	storage, err := sqlite.InMemory()
	if err != nil {
		log.Fatalf("Failed to create SQLite storage: %v", err)
	}

	// Initialize LLM for observation extraction and agent reasoning
	// In production, use real Gemini. For testing, a fake LLM is used.
	modelLLM, err := createLLM(ctx)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Create memory kit with all components (ADK-Go v1.2.0 pattern)
	kit, err := memory.NewMemoryKit(memory.MemoryKitConfig{
		Storage: storage,
		Deriver: memory.NewDeriver(memory.DeriverConfig{
			LLM:     modelLLM,
			Storage: storage,
		}),
	})
	if err != nil {
		log.Fatalf("Failed to create memory kit: %v", err)
	}
	defer kit.Close()

	// Create agent with memory tools for automatic and explicit memory access
	agentInst, err := llmagent.New(llmagent.Config{
		Name:        "memory_assistant",
		Model:       modelLLM,
		Description: "An assistant that remembers facts about users",
		Instruction: "You are a helpful assistant with access to memory. " +
			"Relevant memories are automatically preloaded for each request. " +
			"If the preloaded context is not enough, use the search_memory tool " +
			"to search for additional information. Use what you remember to personalize responses.",
		Tools: []tool.Tool{kit.PreloadTool, kit.LoadTool},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// Create runner with memory service
	appName := "memory_app"
	sessionService := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:           appName,
		Agent:             agentInst,
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

	fmt.Println("Memory-enabled Assistant")
	fmt.Println("Type 'quit' to exit, 'remember' to see what I remember about you.")
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

		// Run the agent with user input
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
				}
			}
		}
		fmt.Println()
	}
}

// createLLM initializes the LLM. In production, uses Gemini API.
// For tests, a mock can be injected.
func createLLM(ctx context.Context) (model.LLM, error) {
	// Check if GOOGLE_API_KEY is set for real LLM
	if os.Getenv("GOOGLE_API_KEY") != "" {
		return gemini.NewModel(ctx, "gemini-2.5-flash", &genai.ClientConfig{
			APIKey: os.Getenv("GOOGLE_API_KEY"),
		})
	}

	// Return nil - tests will use fake LLM
	return nil, fmt.Errorf("GOOGLE_API_KEY not set - use fake LLM for testing")
}

// showMemory displays what the system remembers about the user.
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
		fmt.Println("I don't remember anything about you yet.")
		return
	}

	fmt.Println("Here's what I remember about you:")
	for _, r := range results {
		fmt.Printf("  - %s (confidence: %.2f)\n", r.Observation.Content, r.Score)
	}
}
