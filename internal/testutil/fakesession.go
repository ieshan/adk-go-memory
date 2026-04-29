// Package testutil provides testing utilities for adk-go-memory.
package testutil

import (
	"iter"
	"sync"
	"time"

	"google.golang.org/adk/session"
)

// FakeSession implements session.Session for testing.
type FakeSession struct {
	IDVal      string
	UserIDVal  string
	AppNameVal string
	StateVal   session.State
	EventsVal  []*session.Event
	LastUpdate time.Time
}

// ID implements session.Session.
func (f *FakeSession) ID() string { return f.IDVal }

// UserID implements session.Session.
func (f *FakeSession) UserID() string { return f.UserIDVal }

// AppName implements session.Session.
func (f *FakeSession) AppName() string { return f.AppNameVal }

// State implements session.Session.
func (f *FakeSession) State() session.State {
	if f.StateVal == nil {
		f.StateVal = NewFakeState()
	}
	return f.StateVal
}

// Events implements session.Session.
func (f *FakeSession) Events() session.Events {
	return NewFakeEvents(f.EventsVal)
}

// LastUpdateTime implements session.Session.
func (f *FakeSession) LastUpdateTime() time.Time {
	if f.LastUpdate.IsZero() {
		return time.Now()
	}
	return f.LastUpdate
}

// FakeState implements session.State for testing.
type FakeState struct {
	Mu   sync.RWMutex
	Data map[string]any
}

// NewFakeState creates a new FakeState with initialized data map.
func NewFakeState() *FakeState {
	return &FakeState{Data: make(map[string]any)}
}

// Get implements session.State.
func (f *FakeState) Get(key string) (any, error) {
	f.Mu.RLock()
	defer f.Mu.RUnlock()
	val, ok := f.Data[key]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return val, nil
}

// Set implements session.State.
func (f *FakeState) Set(key string, value any) error {
	f.Mu.Lock()
	defer f.Mu.Unlock()
	if f.Data == nil {
		f.Data = make(map[string]any)
	}
	f.Data[key] = value
	return nil
}

// All implements session.State.
func (f *FakeState) All() iter.Seq2[string, any] {
	f.Mu.RLock()
	defer f.Mu.RUnlock()
	return func(yield func(string, any) bool) {
		for k, v := range f.Data {
			if !yield(k, v) {
				return
			}
		}
	}
}

// FakeEvents implements session.Events for testing.
type FakeEvents struct {
	Events []*session.Event
}

// NewFakeEvents creates a new FakeEvents wrapper.
func NewFakeEvents(events []*session.Event) *FakeEvents {
	return &FakeEvents{Events: events}
}

// All implements session.Events.
func (f *FakeEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, e := range f.Events {
			if !yield(e) {
				return
			}
		}
	}
}

// Len implements session.Events.
func (f *FakeEvents) Len() int { return len(f.Events) }

// At implements session.Events.
func (f *FakeEvents) At(i int) *session.Event {
	if i >= 0 && i < len(f.Events) {
		return f.Events[i]
	}
	return nil
}

// FakeSessionWithEvents is a convenience type for sessions with specific events.
type FakeSessionWithEvents struct {
	FakeSession
	EventsData []*session.Event
}

// Events implements session.Session, returning the specific events.
func (f *FakeSessionWithEvents) Events() session.Events {
	return NewFakeEvents(f.EventsData)
}

// FakeSessionWithState is a convenience type for sessions with specific state.
type FakeSessionWithState struct {
	FakeSession
	StateKey   string
	StateValue any
}

// State implements session.Session, returning state with pre-configured value.
func (f *FakeSessionWithState) State() session.State {
	s := NewFakeState()
	if f.StateValue != nil {
		key := f.StateKey
		if key == "" {
			key = "test_state"
		}
		s.Set(key, f.StateValue)
	}
	return s
}
