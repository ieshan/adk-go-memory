// Package testutil provides testing utilities for adk-go-memory examples.
package testutil

import (
	"context"
	"fmt"
	"iter"

	"google.golang.org/adk/model"
)

// FakeLLM implements model.LLM for deterministic testing without external API calls.
type FakeLLM struct {
	Responses []model.LLMResponse
	Calls     []*model.LLMRequest
}

// Name implements model.LLM.
func (f *FakeLLM) Name() string {
	return "fake-llm"
}

// GenerateContent implements model.LLM.
// It records the request and yields pre-configured responses in order.
func (f *FakeLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	f.Calls = append(f.Calls, req)
	return func(yield func(*model.LLMResponse, error) bool) {
		if len(f.Responses) == 0 {
			yield(nil, fmt.Errorf("fakeLLM: no responses configured"))
			return
		}
		resp := &f.Responses[0]
		f.Responses = f.Responses[1:]
		yield(resp, nil)
	}
}
