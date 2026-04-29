// Package main demonstrates an agent with an explicit memory search tool.
//
// This example shows how to:
//   - Create a memory search tool using NewMemoryTool
//   - Register the tool with an LLM agent
//   - Allow the agent to explicitly query its memory during conversations
//
// The agent can call the search_memory tool with a query to find relevant
// observations from previous conversations.
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
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

func main() {
	ctx := context.Background()

	// Setup storage and memory service
	storage, err := sqlite.InMemory()
	if err != nil {
		log.Fatalf("Failed to create SQLite storage: %v", err)
	}

	modelLLM := getLLM()

	// Create memory kit with all components and tools
	kit, err := memory.New(memory.KitConfig{
		Storage: storage,
		LLM:     modelLLM,
	})
	if err != nil {
		log.Fatalf("Failed to create memory kit: %v", err)
	}
	defer kit.Close()

	// Example with compaction enabled (uncomment to use):
	/*
		kit, err = memory.New(memory.KitConfig{
			Storage: storage,
			LLM:     modelLLM,
			Compaction: &compaction.Config{
				Strategy:   &compaction.SummarizationStrategy{LLM: modelLLM},
				MaxEvents:  50,
				MaxTokens:  2000,
				KeepRecent: 10,
			},
			DeltaMode: true,
		})
		if err != nil {
			log.Fatalf("Failed to create memory kit with compaction: %v", err)
		}
		defer kit.Close()
	*/

	// Create agent with memory tool
	agent, err := llmagent.New(llmagent.Config{
		Name:        "tool_enabled_assistant",
		Model:       modelLLM,
		Description: "Assistant with memory search capability",
		Instruction: `You are a helpful assistant with access to a memory tool.

When users ask about their preferences, past activities, or personal information,
use the search_memory tool to find relevant information from previous conversations.

To use the tool, call search_memory with a query describing what you're looking for.
Example queries: "user preferences", "user name", "user hobbies", "past activities"`,
		Tools: []tool.Tool{kit.LoadTool},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// Create runner
	// With compaction plugin (requires kit.Plugin to be set):
	// runner.New(runner.Config{
	//     AppName:           appName,
	//     Agent:             agent,
	//     SessionService:    sessionService,
	//     MemoryService:     kit.Service,
	//     AutoCreateSession: true,
	//     PluginConfig: runner.PluginConfig{
	//         Plugins: []*plugin.Plugin{kit.Plugin},
	//     },
	// })
	appName := "memory_tool_app"
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

	fmt.Println("Memory Tool Demo")
	fmt.Println("The assistant can explicitly search its memory using a tool.")
	fmt.Println("Type 'quit' to exit, 'remember' to see stored memories.")
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
						fmt.Printf("\n[Tool Call: %s(%v)]\n", part.FunctionCall.Name, part.FunctionCall.Args)
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
		fmt.Printf("  - %s (level: %s, score: %.2f)\n",
			r.Observation.Content, r.Observation.Level, r.Score)
	}
}
