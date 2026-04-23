// Package main demonstrates a basic agent with memory integration.
//
// This example shows how to:
//   - Set up an in-memory SQLite storage for observations
//   - Create a Deriver for automatic fact extraction from conversations
//   - Wire the memory service into an ADK-Go agent
//   - Use BeforeModelCallback to inject memory context into prompts
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

	"google.golang.org/adk/agent"
	adkmemory "google.golang.org/adk/memory"

	memory "github.com/ieshan/adk-go-memory"
	"github.com/ieshan/adk-go-memory/adapter"
	adkagent "google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

func main() {
	ctx := context.Background()

	// Setup in-memory SQLite storage
	storage, err := adapter.InMemory()
	if err != nil {
		log.Fatalf("Failed to create storage: %v", err)
	}
	defer storage.Close()

	// Initialize LLM for observation extraction and agent reasoning
	// In production, use real Gemini. For testing, a fake LLM is used.
	modelLLM, err := createLLM(ctx)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Create deriver for automatic fact extraction from conversations
	deriver := memory.NewDeriver(memory.DeriverConfig{
		LLM:     modelLLM,
		Storage: storage,
	})

	// Create memory service implementing google.golang.org/adk/memory.Service
	svc := memory.NewService(memory.ServiceConfig{
		Storage: storage,
		Deriver: deriver,
	})
	defer svc.Close()

	// Create agent with memory-aware instruction and callback
	agentInst, err := llmagent.New(llmagent.Config{
		Name:        "memory_assistant",
		Model:       modelLLM,
		Description: "An assistant that remembers facts about users",
		Instruction: "You are a helpful assistant. Use the conversation context to answer questions. " +
			"If the user mentions facts about themselves, acknowledge them. " +
			"If you have context from memory, use it to personalize your responses.",
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{
			injectMemoryContext(svc),
		},
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
		MemoryService:     svc,
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

// injectMemoryContext returns a BeforeModelCallback that searches memory
// and injects relevant context into the system prompt.
func injectMemoryContext(svc *memory.Service) llmagent.BeforeModelCallback {
	return func(ctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
		// Get the user's query from the request
		if len(req.Contents) == 0 {
			return nil, nil
		}

		// Find the most recent user content
		var userQuery string
		for i := len(req.Contents) - 1; i >= 0; i-- {
			content := req.Contents[i]
			if content.Role == genai.RoleUser && len(content.Parts) > 0 {
				userQuery = content.Parts[0].Text
				break
			}
		}

		if userQuery == "" {
			return nil, nil
		}

		// Search memory for relevant context
		searchResp, err := svc.SearchMemory(ctx, &adkmemory.SearchRequest{
			Query:   userQuery,
			UserID:  ctx.UserID(),
			AppName: ctx.AppName(),
		})
		if err != nil {
			// Log but don't fail the request
			log.Printf("Memory search error: %v", err)
			return nil, nil
		}

		if len(searchResp.Memories) == 0 {
			return nil, nil
		}

		// Build memory context string
		var memContext strings.Builder
		memContext.WriteString("\n\n**Relevant Information from Memory:**\n")
		for _, mem := range searchResp.Memories {
			if mem.Content != nil && len(mem.Content.Parts) > 0 {
				memContext.WriteString("- ")
				memContext.WriteString(mem.Content.Parts[0].Text)
				memContext.WriteString("\n")
			}
		}

		// Inject memory into system instruction (first content if it's a system message)
		for _, content := range req.Contents {
			if content.Role == "system" || content.Role == genai.RoleModel {
				if len(content.Parts) > 0 {
					content.Parts[0].Text += memContext.String()
					break
				}
			}
		}

		return nil, nil
	}
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
