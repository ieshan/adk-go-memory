package memory

import (
	"fmt"
	"strings"
	"time"

	"github.com/ieshan/adk-go-memory/adapter"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

const preloadInstructions = `The following content is from your previous conversations with the user.
They may be useful for answering the user's current query.
<PAST_CONVERSATIONS>
%s
</PAST_CONVERSATIONS>`

// PreloadMemoryTool implements ONLY RequestProcessor (no FunctionTool).
// It automatically preloads relevant memory context into system instructions
// before each LLM request, without requiring the model to call a function.
type PreloadMemoryTool struct {
	provider *Provider
}

// NewPreloadMemoryTool creates a new preload memory tool wired to a Provider.
func NewPreloadMemoryTool(provider *Provider) *PreloadMemoryTool {
	return &PreloadMemoryTool{provider: provider}
}

// Name implements tool.Tool.
func (t *PreloadMemoryTool) Name() string {
	return "preload_memory"
}

// Description implements tool.Tool.
func (t *PreloadMemoryTool) Description() string {
	return "Preloads relevant memory for the current user."
}

// IsLongRunning implements tool.Tool.
func (t *PreloadMemoryTool) IsLongRunning() bool {
	return false
}

// ProcessRequest implements toolinternal.RequestProcessor.
// It searches memory using the user's current query and injects relevant
// past conversations into system instructions.
func (t *PreloadMemoryTool) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	// Get the user's current query from context
	userContent := ctx.UserContent()
	if userContent == nil || len(userContent.Parts) == 0 || userContent.Parts[0].Text == "" {
		return nil
	}
	userQuery := userContent.Parts[0].Text

	// Search for relevant memories
	observations, err := t.provider.SearchMemory(ctx, userQuery, "", ctx.UserID(), ctx.AppName())
	if err != nil {
		return fmt.Errorf("preload memory: search failed: %w", err)
	}
	if len(observations) == 0 {
		return nil
	}

	// Format memories and inject into instructions
	memoryText := formatPreloadMemories(observations)
	if memoryText == "" {
		return nil
	}

	// Append to system instructions
	appendInstructions(req, fmt.Sprintf(preloadInstructions, memoryText))
	return nil
}

// formatPreloadMemories formats observations for injection into system instructions.
func formatPreloadMemories(observations []adapter.Observation) string {
	var lines []string
	for _, obs := range observations {
		if obs.Content == "" {
			continue
		}

		// Format with timestamp if available
		if !obs.CreatedAt.IsZero() {
			lines = append(lines, fmt.Sprintf("Time: %s", obs.CreatedAt.Format(time.RFC3339)))
		}

		// Include level indicator for context
		content := fmt.Sprintf("[%s] %s", obs.Level, obs.Content)
		lines = append(lines, content)
	}
	return strings.Join(lines, "\n")
}

// appendInstructions appends text to the system instruction in the request.
// Similar to google.golang.org/adk/internal/utils.AppendInstructions
func appendInstructions(req *model.LLMRequest, text string) {
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if req.Config.SystemInstruction == nil {
		req.Config.SystemInstruction = &genai.Content{
			Role: "system",
		}
	}

	// Find or create text part
	found := false
	for _, part := range req.Config.SystemInstruction.Parts {
		if part != nil && part.Text != "" {
			part.Text += "\n\n" + text
			found = true
			break
		}
	}

	if !found {
		req.Config.SystemInstruction.Parts = append(
			req.Config.SystemInstruction.Parts,
			&genai.Part{Text: text},
		)
	}
}
