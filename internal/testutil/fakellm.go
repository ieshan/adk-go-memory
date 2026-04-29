// Package testutil provides testing utilities for adk-go-memory.
package testutil

import (
	"context"
	"fmt"
	"iter"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// FakeLLM implements model.LLM for deterministic testing without external API calls.
type FakeLLM struct {
	Responses []model.LLMResponse
	Calls     []*model.LLMRequest
	Err       error // If set, yields this error instead of Responses
}

// Name implements model.LLM.
func (f *FakeLLM) Name() string {
	return "fake-llm"
}

// GenerateContent implements model.LLM.
// It records the request and yields pre-configured responses in order.
// If Err is set, yields the error instead.
func (f *FakeLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	f.Calls = append(f.Calls, req)
	return func(yield func(*model.LLMResponse, error) bool) {
		if f.Err != nil {
			yield(nil, f.Err)
			return
		}
		if len(f.Responses) == 0 {
			yield(nil, fmt.Errorf("fakeLLM: no responses configured"))
			return
		}
		resp := &f.Responses[0]
		f.Responses = f.Responses[1:]
		yield(resp, nil)
	}
}

// FakeGenaiLLM implements the genai client interface used by compaction strategies.
// This is a simplified interface for the Google GenAI client.
type FakeGenaiLLM struct {
	Response   string
	AllowEmpty bool
	Err        error
}

// GenerateContent implements the genai LLM interface.
func (f *FakeGenaiLLM) GenerateContent(ctx context.Context, contents ...*genai.Content) (*genai.GenerateContentResponse, error) {
	if f.Err != nil {
		return nil, f.Err
	}
	text := f.Response
	if text == "" && !f.AllowEmpty {
		text = "Summary of conversation"
	}
	return &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{
				Content: genai.NewContentFromText(text, genai.RoleModel),
			},
		},
	}, nil
}

// ErrorGenaiLLM always returns an error.
type ErrorGenaiLLM struct {
	Err error
}

// GenerateContent implements the genai LLM interface, always returning an error.
func (e *ErrorGenaiLLM) GenerateContent(ctx context.Context, contents ...*genai.Content) (*genai.GenerateContentResponse, error) {
	if e.Err != nil {
		return nil, e.Err
	}
	return nil, fmt.Errorf("LLM unavailable")
}
