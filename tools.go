package memory

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/ieshan/adk-go-memory/adapter"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

// SearchMemoryArgs matches the tool schema expected by the LLM.
// Query is the search query string. MaxResults optionally limits the number of results (default: 10).
type SearchMemoryArgs struct {
	Query      string `json:"query"`
	MaxResults int    `json:"max_results,omitempty"`
}

// SearchMemoryResults returned by the tool.
// Observations contains the matching memory entries found by the search.
type SearchMemoryResults struct {
	Observations []ToolObservation `json:"observations"`
}

// ToolObservation represents a single observation in tool results.
// Content is the observation text. Level indicates confidence (explicit, deductive, inductive, contradiction).
// Tags are optional category labels associated with this observation.
type ToolObservation struct {
	Content string   `json:"content"`
	Level   string   `json:"level"`
	Tags    []string `json:"tags,omitempty"`
}

// MemoryTool implements the tool interface for memory search.
type MemoryTool struct {
	storage adapter.Storage
}

// NewMemoryTool creates a new memory search tool.
func NewMemoryTool(storage adapter.Storage) (*MemoryTool, error) {
	return &MemoryTool{storage: storage}, nil
}

// Name returns the name of the tool.
func (m *MemoryTool) Name() string {
	return "search_memory"
}

// Description returns a description of the tool.
func (m *MemoryTool) Description() string {
	return "Search the agent's memory for relevant observations about the user."
}

// IsLongRunning indicates whether the tool is a long-running operation.
func (m *MemoryTool) IsLongRunning() bool {
	return false
}

// Declaration returns the function declaration for the LLM.
func (m *MemoryTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        m.Name(),
		Description: m.Description(),
		Parameters: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"query": {
					Type:        genai.TypeString,
					Description: "The search query to find relevant memories",
				},
				"max_results": {
					Type:        genai.TypeInteger,
					Description: "Maximum number of results to return (default: 10)",
				},
			},
			Required: []string{"query"},
		},
	}
}

// Call executes the memory search.
func (m *MemoryTool) Call(ctx context.Context, argsJSON string) (string, error) {
	var args SearchMemoryArgs
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("memory tool: parse args: %w", err)
	}

	maxResults := args.MaxResults
	if maxResults == 0 {
		maxResults = 10
	}

	results, err := m.storage.Search(ctx, &adapter.SearchOptions{
		Query:      args.Query,
		MaxResults: maxResults,
		Mode:       adapter.SearchModeHybrid,
	})
	if err != nil {
		return "", fmt.Errorf("memory tool: search: %w", err)
	}

	output := SearchMemoryResults{
		Observations: make([]ToolObservation, len(results)),
	}
	for i, r := range results {
		output.Observations[i] = ToolObservation{
			Content: r.Observation.Content,
			Level:   string(r.Observation.Level),
			Tags:    r.Observation.Tags,
		}
	}

	outputJSON, err := json.Marshal(output)
	if err != nil {
		return "", fmt.Errorf("memory tool: marshal results: %w", err)
	}

	return string(outputJSON), nil
}

// ProcessRequest implements toolinternal.RequestProcessor.
// It packs the tool declaration into the LLM request.
func (m *MemoryTool) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	// Pack the tool declaration into the request
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	decl := m.Declaration()
	req.Config.Tools = append(req.Config.Tools, &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{decl},
	})
	return nil
}
