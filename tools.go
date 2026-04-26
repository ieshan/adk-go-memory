package memory

import (
	"fmt"

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

// MemoryTool implements toolinternal.FunctionTool and RequestProcessor for memory search.
type MemoryTool struct {
	provider *Provider
}

// NewMemoryTool creates a new memory search tool wired to a Provider.
func NewMemoryTool(provider *Provider) *MemoryTool {
	return &MemoryTool{provider: provider}
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

// Run executes the memory search (implements toolinternal.FunctionTool).
func (m *MemoryTool) Run(ctx tool.Context, args any) (map[string]any, error) {
	margs, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("memory tool: invalid args type %T", args)
	}

	queryRaw, ok := margs["query"]
	if !ok {
		return nil, fmt.Errorf("memory tool: missing required parameter: query")
	}
	query, ok := queryRaw.(string)
	if !ok {
		return nil, fmt.Errorf("memory tool: query must be a string, got %T", queryRaw)
	}

	maxResults := 10
	if maxRaw, ok := margs["max_results"]; ok {
		if maxFloat, ok := maxRaw.(float64); ok {
			maxResults = int(maxFloat)
		}
	}

	// Use Provider for comprehensive search (includes representation manager)
	observations, err := m.provider.SearchMemory(ctx, query, "", ctx.UserID(), ctx.AppName())
	if err != nil {
		return nil, fmt.Errorf("memory tool: search: %w", err)
	}

	// Limit results
	if len(observations) > maxResults {
		observations = observations[:maxResults]
	}

	output := SearchMemoryResults{
		Observations: make([]ToolObservation, len(observations)),
	}
	for i, obs := range observations {
		output.Observations[i] = ToolObservation{
			Content: obs.Content,
			Level:   string(obs.Level),
			Tags:    obs.Tags,
		}
	}

	return map[string]any{
		"observations": output.Observations,
	}, nil
}

// ProcessRequest implements toolinternal.RequestProcessor.
// It packs the tool declaration into the LLM request.
func (m *MemoryTool) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	if req.Tools == nil {
		req.Tools = make(map[string]any)
	}
	req.Tools[m.Name()] = m

	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}

	// Find existing tool with FunctionDeclarations or create new one
	var funcTool *genai.Tool
	for _, tool := range req.Config.Tools {
		if tool != nil && tool.FunctionDeclarations != nil {
			funcTool = tool
			break
		}
	}
	if funcTool == nil {
		req.Config.Tools = append(req.Config.Tools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{m.Declaration()},
		})
	} else {
		funcTool.FunctionDeclarations = append(funcTool.FunctionDeclarations, m.Declaration())
	}
	return nil
}
