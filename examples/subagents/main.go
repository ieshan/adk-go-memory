// Package main demonstrates an agent with sub-agents sharing memory.
//
// This example shows how to:
//   - Create a parent agent with specialized sub-agents
//   - Share memory across the agent tree via runner configuration
//   - Delegate tasks from parent to sub-agents
//
// The parent agent can delegate to either:
//   - fact_recorder: Records and acknowledges user facts
//   - greeter: Greets users by name using memory context
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/genai"

	adkmemory "google.golang.org/adk/memory"

	memory "github.com/ieshan/adk-go-memory"
	"github.com/ieshan/adk-go-memory/adapter"
	adkagent "google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

func main() {
	ctx := context.Background()

	// Setup in-memory storage and memory service
	storage, err := adapter.InMemory()
	if err != nil {
		log.Fatalf("Failed to create storage: %v", err)
	}
	defer storage.Close()

	// For production, use a real LLM. For this example, we'll need an LLM.
	modelLLM := getLLM()

	// Create deriver for automatic fact extraction
	deriver := memory.NewDeriver(memory.DeriverConfig{
		LLM:     modelLLM,
		Storage: storage,
	})

	// Create memory service
	svc := memory.NewService(memory.ServiceConfig{
		Storage: storage,
		Deriver: deriver,
	})
	defer svc.Close()

	// Create sub-agents
	factRecorder := createFactRecorder(modelLLM)
	greeter := createGreeter(modelLLM, svc)

	// Create parent agent with sub-agents
	parentAgent, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Model:       modelLLM,
		Description: "Main assistant that can record facts or greet users. Delegate to fact_recorder for recording facts, greeter for greetings.",
		Instruction: "You are a helpful assistant with two specialized sub-agents:\n" +
			"1. fact_recorder - Use when users share facts about themselves\n" +
			"2. greeter - Use for greetings and social interactions\n" +
			"\nDelegate to the appropriate sub-agent based on the user's message.",
		SubAgents: []adkagent.Agent{factRecorder, greeter},
	})
	if err != nil {
		log.Fatalf("Failed to create parent agent: %v", err)
	}

	// Create runner with shared memory service
	appName := "subagent_app"
	sessionService := session.InMemoryService()
	r, err := runner.New(runner.Config{
		AppName:           appName,
		Agent:             parentAgent,
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

	fmt.Println("Sub-Agent Memory Demo")
	fmt.Println("Type 'quit' to exit, 'remember' to see what I remember.")
	fmt.Println("Try: 'My name is Alice' then 'Say hello' to see memory in action.")
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
				}
			}
		}
		fmt.Println()
	}
}

// createFactRecorder creates a sub-agent that records user facts.
func createFactRecorder(modelLLM model.LLM) adkagent.Agent {
	agent, err := llmagent.New(llmagent.Config{
		Name:        "fact_recorder",
		Model:       modelLLM,
		Description: "Records and acknowledges user facts.",
		Instruction: "When users share facts about themselves, acknowledge and confirm you've noted them. " +
			"Be warm and encouraging about sharing.",
	})
	if err != nil {
		log.Fatalf("Failed to create fact_recorder: %v", err)
	}
	return agent
}

// createGreeter creates a sub-agent that greets users using memory context.
func createGreeter(modelLLM model.LLM, svc *memory.Service) adkagent.Agent {
	agent, err := llmagent.New(llmagent.Config{
		Name:        "greeter",
		Model:       modelLLM,
		Description: "Greets users by name using memory context.",
		Instruction: "Greet the user warmly. If you know their name from context, use it! " +
			"Be friendly and personable.",
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{
			func(ctx adkagent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
				// Search memory for user name
				searchResp, err := svc.SearchMemory(ctx, &adkmemory.SearchRequest{
					Query:   "user name",
					UserID:  ctx.UserID(),
					AppName: ctx.AppName(),
				})
				if err != nil || len(searchResp.Memories) == 0 {
					return nil, nil
				}

				// Inject memory into system context
				memContext := "\n\n**Known Facts:**\n"
				for _, mem := range searchResp.Memories {
					if mem.Content != nil && len(mem.Content.Parts) > 0 {
						memContext += "- " + mem.Content.Parts[0].Text + "\n"
					}
				}

				// Find system content and append
				for _, content := range req.Contents {
					if content.Role == "system" || content.Role == genai.RoleModel {
						if len(content.Parts) > 0 {
							content.Parts[0].Text += memContext
							break
						}
					}
				}
				return nil, nil
			},
		},
	})
	if err != nil {
		log.Fatalf("Failed to create greeter: %v", err)
	}
	return agent
}

// getLLM returns the LLM to use. In production, this would use Gemini.
func getLLM() model.LLM {
	// Return nil - this is a demonstration
	// Production code would use gemini.NewModel()
	return nil
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
