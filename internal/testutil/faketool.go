// Package testutil provides testing utilities for adk-go-memory.
package testutil

import (
	"context"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool/toolconfirmation"
	"google.golang.org/genai"
)

// FakeToolContext implements tool.Context for testing.
type FakeToolContext struct {
	UserIDVal    string
	AppNameVal   string
	SessionIDVal string
	AgentNameVal string
	StateVal     session.State
	Ctx          context.Context
}

// NewFakeToolContext creates a new FakeToolContext with defaults.
func NewFakeToolContext(userID, appName string) *FakeToolContext {
	return &FakeToolContext{
		UserIDVal:  userID,
		AppNameVal: appName,
		Ctx:        context.Background(),
	}
}

// FunctionCallID implements tool.Context.
func (f *FakeToolContext) FunctionCallID() string { return "" }

// Actions implements tool.Context.
func (f *FakeToolContext) Actions() *session.EventActions { return &session.EventActions{} }

// SearchMemory implements tool.Context.
func (f *FakeToolContext) SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error) {
	return nil, nil
}

// ToolConfirmation implements tool.Context.
func (f *FakeToolContext) ToolConfirmation() *toolconfirmation.ToolConfirmation { return nil }

// RequestConfirmation implements tool.Context.
func (f *FakeToolContext) RequestConfirmation(hint string, payload any) error { return nil }

// UserID implements tool.Context.
func (f *FakeToolContext) UserID() string { return f.UserIDVal }

// AppName implements tool.Context.
func (f *FakeToolContext) AppName() string { return f.AppNameVal }

// SessionID implements tool.Context.
func (f *FakeToolContext) SessionID() string { return f.SessionIDVal }

// AgentName implements tool.Context.
func (f *FakeToolContext) AgentName() string { return f.AgentNameVal }

// State implements tool.Context.
func (f *FakeToolContext) State() session.State {
	if f.StateVal == nil {
		return nil
	}
	return f.StateVal
}

// Artifacts implements tool.Context.
func (f *FakeToolContext) Artifacts() agent.Artifacts { return nil }

// InvocationContext implements tool.Context.
func (f *FakeToolContext) InvocationContext() agent.InvocationContext { return nil }

// EndInvocation implements tool.Context.
func (f *FakeToolContext) EndInvocation() {}

// Ended implements tool.Context.
func (f *FakeToolContext) Ended() bool { return false }

// UserContent implements tool.Context.
func (f *FakeToolContext) UserContent() *genai.Content {
	return &genai.Content{Parts: []*genai.Part{{Text: ""}}}
}

// Context implements tool.Context (embeds context.Context).
func (f *FakeToolContext) Context() context.Context {
	if f.Ctx == nil {
		return context.Background()
	}
	return f.Ctx
}

// Branch implements tool.Context.
func (f *FakeToolContext) Branch() string { return "" }

// InvocationID implements tool.Context.
func (f *FakeToolContext) InvocationID() string { return "" }

// ReadonlyState implements tool.Context.
func (f *FakeToolContext) ReadonlyState() session.ReadonlyState { return nil }

// Deadline implements context.Context.
func (f *FakeToolContext) Deadline() (deadline time.Time, ok bool) { return time.Time{}, false }

// Done implements context.Context.
func (f *FakeToolContext) Done() <-chan struct{} { return nil }

// Err implements context.Context.
func (f *FakeToolContext) Err() error { return nil }

// Value implements context.Context.
func (f *FakeToolContext) Value(key any) any { return nil }
