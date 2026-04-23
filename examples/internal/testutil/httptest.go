// Package testutil provides HTTP testing utilities for adk-go-memory examples.
package testutil

import (
	"fmt"
	"net/http"
	"strings"
)

// MockTransport is an http.RoundTripper that redirects requests to a mock server.
type MockTransport struct {
	BaseURL string
}

// RoundTrip implements http.RoundTripper.
// It replaces the request URL host with the mock server base URL.
func (m *MockTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Replace the host with the mock server
	mockURL := m.BaseURL + req.URL.Path
	if req.URL.RawQuery != "" {
		mockURL = mockURL + "?" + req.URL.RawQuery
	}

	// Create new request with mock URL
	newReq, err := http.NewRequest(req.Method, mockURL, req.Body)
	if err != nil {
		return nil, fmt.Errorf("mock transport: create request: %w", err)
	}
	newReq.Header = req.Header

	// Use default transport to actually make the request to mock server
	return http.DefaultTransport.RoundTrip(newReq)
}

// RewriteURL replaces a base URL in a string with another base URL.
// Useful for normalizing URLs in test assertions.
func RewriteURL(original, oldBase, newBase string) string {
	return strings.Replace(original, oldBase, newBase, -1)
}
