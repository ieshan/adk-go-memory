package compaction

import (
	"testing"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestNewPlugin_ValidConfig(t *testing.T) {
	strategy := &TruncationStrategy{RetainCount: 10}
	cfg := &Config{
		Strategy:   strategy,
		MaxEvents:  100,
		KeepRecent: 10,
	}

	plugin, err := NewPlugin(cfg)
	if err != nil {
		t.Fatalf("NewPlugin() error = %v", err)
	}
	if plugin == nil {
		t.Fatal("NewPlugin() returned nil plugin")
	}
}

func TestNewPlugin_InvalidConfig(t *testing.T) {
	strategy := &TruncationStrategy{RetainCount: 10}
	cfg := &Config{
		Strategy:   strategy,
		KeepRecent: 10,
		// No MaxEvents, MaxTokens, or Interval - should fail
	}

	_, err := NewPlugin(cfg)
	if err == nil {
		t.Fatal("NewPlugin() expected error for invalid config, got nil")
	}
}

func TestBeforeModelCallback_ValidConfig(t *testing.T) {
	strategy := &TruncationStrategy{RetainCount: 10}
	cfg := &Config{
		Strategy:   strategy,
		MaxEvents:  100,
		KeepRecent: 10,
	}

	callback, err := BeforeModelCallback(cfg)
	if err != nil {
		t.Fatalf("BeforeModelCallback() error = %v", err)
	}
	if callback == nil {
		t.Fatal("BeforeModelCallback() returned nil")
	}
}

func TestBeforeModelCallback_InvalidConfig(t *testing.T) {
	strategy := &TruncationStrategy{RetainCount: 10}
	cfg := &Config{
		Strategy:   strategy,
		KeepRecent: 10,
		// No triggers
	}

	_, err := BeforeModelCallback(cfg)
	if err == nil {
		t.Fatal("BeforeModelCallback() expected error for invalid config")
	}
}

func TestAfterAgentCallback_ValidConfig(t *testing.T) {
	strategy := &TruncationStrategy{RetainCount: 10}
	cfg := &Config{
		Strategy:         strategy,
		MaxEvents:        100,
		KeepRecent:       10,
		EnableAfterAgent: true,
	}

	callback, err := AfterAgentCallback(cfg)
	if err != nil {
		t.Fatalf("AfterAgentCallback() error = %v", err)
	}
	if callback == nil {
		t.Fatal("AfterAgentCallback() returned nil")
	}
}

func TestAfterAgentCallback_InvalidConfig(t *testing.T) {
	strategy := &TruncationStrategy{RetainCount: 10}
	cfg := &Config{
		Strategy:   strategy,
		KeepRecent: 10,
		// No triggers
	}

	_, err := AfterAgentCallback(cfg)
	if err == nil {
		t.Fatal("AfterAgentCallback() expected error for invalid config")
	}
}

func TestBeforeAgentCallback_ValidConfig(t *testing.T) {
	strategy := &TruncationStrategy{RetainCount: 10}
	cfg := &Config{
		Strategy:   strategy,
		MaxEvents:  100,
		KeepRecent: 10,
	}

	callback, err := BeforeAgentCallback(cfg)
	if err != nil {
		t.Fatalf("BeforeAgentCallback() error = %v", err)
	}
	if callback == nil {
		t.Fatal("BeforeAgentCallback() returned nil")
	}
}

func TestBeforeAgentCallback_InvalidConfig(t *testing.T) {
	strategy := &TruncationStrategy{RetainCount: 10}
	cfg := &Config{
		Strategy:   strategy,
		KeepRecent: 10,
		// No triggers
	}

	_, err := BeforeAgentCallback(cfg)
	if err == nil {
		t.Fatal("BeforeAgentCallback() expected error for invalid config")
	}
}

func TestHandler_InjectSummary(t *testing.T) {
	strategy := &TruncationStrategy{RetainCount: 10}
	cfg := &Config{
		Strategy:   strategy,
		MaxEvents:  100,
		KeepRecent: 10,
	}

	h, err := newHandler(cfg)
	if err != nil {
		t.Fatalf("newHandler() error = %v", err)
	}

	// Test with nil request - should not panic
	h.injectSummary(nil, "summary")

	// Test with empty summary - should not modify request
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("test", genai.RoleUser),
		},
	}
	originalLen := len(req.Contents)
	h.injectSummary(req, "")
	if len(req.Contents) != originalLen {
		t.Error("injectSummary with empty text should not modify request")
	}

	// Test actual injection
	h.injectSummary(req, "Test summary")
	if len(req.Contents) != originalLen+1 {
		t.Errorf("Expected %d contents after injection, got %d", originalLen+1, len(req.Contents))
	}
	if len(req.Contents) > 0 {
		expected := "[Previous conversation summarized]: Test summary"
		if req.Contents[0].Parts[0].Text != expected {
			t.Errorf("Summary not properly injected: got %v, want %v", req.Contents[0].Parts[0].Text, expected)
		}
	}
}
