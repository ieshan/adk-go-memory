package compaction

import (
	"fmt"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/plugin"
	"google.golang.org/genai"
)

// handler implements the core compaction logic shared between Plugin and standalone callbacks.
type handler struct {
	config *Config
}

// newHandler creates a new compaction handler.
func newHandler(cfg *Config) (*handler, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &handler{config: cfg}, nil
}

// handleCompaction performs the compaction check and returns modified request or nil.
// Note: CallbackContext has State() but not Session(). We track compaction markers in State,
// and the actual event compaction happens in AddSessionToMemory when the Service processes
// the session.
func (h *handler) handleCompaction(ctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
	// Get current compaction state from session.State
	state, err := h.getState(ctx)
	if err != nil {
		return nil, fmt.Errorf("compaction: get state: %w", err)
	}

	// We can't access Session.Events() from CallbackContext, so we use a different approach:
	// 1. Store the request to compact in State
	// 2. The Service reads this marker in AddSessionToMemory and performs actual compaction
	// 3. Inject any previously computed summary into the request

	// Check if there's a pending summary to inject
	summaryText, err := h.getPendingSummary(ctx)
	if err == nil && summaryText != "" {
		h.injectSummary(req, summaryText)
	}

	// Mark that compaction should be checked (Service will handle actual event processing)
	if err := h.markCompactionNeeded(ctx, state); err != nil {
		return nil, fmt.Errorf("compaction: mark compaction: %w", err)
	}

	return nil, nil
}

// injectSummary adds compaction summary to the LLM request.
func (h *handler) injectSummary(req *model.LLMRequest, summaryText string) {
	if req == nil || summaryText == "" {
		return
	}

	// Create summary content
	summaryContent := genai.NewContentFromText(
		fmt.Sprintf("[Previous conversation summarized]: %s", summaryText),
		genai.RoleModel,
	)

	// Prepend summary to contents
	newContents := make([]*genai.Content, 0, len(req.Contents)+1)
	newContents = append(newContents, summaryContent)
	newContents = append(newContents, req.Contents...)
	req.Contents = newContents
}

// getState retrieves compaction state from session.State.
func (h *handler) getState(ctx agent.CallbackContext) (*state, error) {
	value, err := ctx.State().Get(h.config.StateKey())
	if err != nil {
		// State key not found - return zero state
		return &state{}, nil
	}

	s, ok := value.(*state)
	if !ok {
		return nil, fmt.Errorf("compaction: get state: invalid type %T", value)
	}

	return s, nil
}

// getPendingSummary retrieves any pending summary from State.
func (h *handler) getPendingSummary(ctx agent.CallbackContext) (string, error) {
	key := h.config.StateKey() + "_pending_summary"
	value, err := ctx.State().Get(key)
	if err != nil {
		return "", err
	}

	summary, ok := value.(string)
	if !ok {
		return "", fmt.Errorf("compaction: get pending summary: invalid type %T", value)
	}

	// Clear the pending summary after retrieval
	_ = ctx.State().Set(key, "")

	return summary, nil
}

// markCompactionNeeded sets a marker in State that compaction is needed.
func (h *handler) markCompactionNeeded(ctx agent.CallbackContext, st *state) error {
	return ctx.State().Set(h.config.StateKey(), st)
}

// handleAfterAgent performs post-run compaction check.
func (h *handler) handleAfterAgent(ctx agent.CallbackContext) (*genai.Content, error) {
	if !h.config.EnableAfterAgent {
		return nil, nil
	}

	// Re-run compaction check after agent completes
	// This uses empty request since we're not modifying LLM input
	_, err := h.handleCompaction(ctx, &model.LLMRequest{
		Contents: make([]*genai.Content, 0),
	})
	return nil, err
}

// Plugin wraps compaction functionality as an ADK plugin.
type Plugin struct {
	handler *handler
}

// NewPlugin creates a compaction plugin for use with Runner.
//
// Example:
//
//	plugin, err := compaction.NewPlugin(&compaction.Config{
//	    Strategy:   &compaction.SummarizationStrategy{LLM: llm},
//	    MaxEvents:  100,
//	    MaxTokens:  4000,
//	    KeepRecent: 20,
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	runner.New(runner.Config{
//	    PluginConfig: runner.PluginConfig{
//	        Plugins: []*plugin.Plugin{plugin},
//	    },
//	})
func NewPlugin(cfg *Config) (*plugin.Plugin, error) {
	h, err := newHandler(cfg)
	if err != nil {
		return nil, err
	}

	return plugin.New(plugin.Config{
		Name:                "memory-compaction",
		BeforeModelCallback: h.beforeModelCallback(),
		AfterAgentCallback:  h.afterAgentCallback(),
	})
}

// beforeModelCallback returns the BeforeModelCallback function.
func (h *handler) beforeModelCallback() llmagent.BeforeModelCallback {
	return func(ctx agent.CallbackContext, req *model.LLMRequest) (*model.LLMResponse, error) {
		return h.handleCompaction(ctx, req)
	}
}

// afterAgentCallback returns the AfterAgentCallback function.
func (h *handler) afterAgentCallback() agent.AfterAgentCallback {
	return func(ctx agent.CallbackContext) (*genai.Content, error) {
		return h.handleAfterAgent(ctx)
	}
}

// BeforeModelCallback creates a standalone BeforeModelCallback for direct LlmAgent usage.
//
// Use this when using LlmAgent directly without Runner.
//
// Example:
//
//	callback := compaction.BeforeModelCallback(&compaction.Config{
//	    Strategy:   &compaction.TruncationStrategy{RetainCount: 50},
//	    MaxEvents:  100,
//	    KeepRecent: 10,
//	})
//
//	agent, _ := llmagent.New(llmagent.Config{
//	    Model: model,
//	    BeforeModelCallbacks: []llmagent.BeforeModelCallback{callback},
//	})
func BeforeModelCallback(cfg *Config) (llmagent.BeforeModelCallback, error) {
	h, err := newHandler(cfg)
	if err != nil {
		return nil, err
	}
	return h.beforeModelCallback(), nil
}

// AfterAgentCallback creates a standalone AfterAgentCallback for direct LlmAgent usage.
//
// Use this when using LlmAgent directly without Runner and want post-run compaction.
//
// Example:
//
//	callback, _ := compaction.AfterAgentCallback(&compaction.Config{
//	    Strategy:         &compaction.SummarizationStrategy{LLM: llm},
//	    MaxEvents:        100,
//	    EnableAfterAgent: true,
//	})
//
//	agent, _ := llmagent.New(llmagent.Config{
//	    Model: model,
//	    AfterAgentCallbacks: []agent.AfterAgentCallback{callback},
//	})
func AfterAgentCallback(cfg *Config) (agent.AfterAgentCallback, error) {
	h, err := newHandler(cfg)
	if err != nil {
		return nil, err
	}
	return h.afterAgentCallback(), nil
}

// BeforeAgentCallback creates a BeforeAgentCallback for compaction.
// This runs before the agent starts and can be used for early compaction.
//
// Example:
//
//	callback, _ := compaction.BeforeAgentCallback(&compaction.Config{...})
//
//	agent, _ := llmagent.New(llmagent.Config{
//	    Model: model,
//	    BeforeAgentCallbacks: []agent.BeforeAgentCallback{callback},
//	})
func BeforeAgentCallback(cfg *Config) (agent.BeforeAgentCallback, error) {
	h, err := newHandler(cfg)
	if err != nil {
		return nil, err
	}

	return func(ctx agent.CallbackContext) (*genai.Content, error) {
		// BeforeAgentCallback returns Content, not LLMResponse
		// We run compaction but don't return anything (let agent run)
		h.handleCompaction(ctx, &model.LLMRequest{
			Contents: make([]*genai.Content, 0),
		})
		return nil, nil
	}, nil
}
