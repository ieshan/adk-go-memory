// Package testutil provides testing utilities for adk-go-memory.
package testutil

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"time"

	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// CreateEvents creates n session events with generated content.
func CreateEvents(n int) []*session.Event {
	events := make([]*session.Event, n)
	for i := 0; i < n; i++ {
		events[i] = &session.Event{
			ID:        fmt.Sprintf("event-%d", i),
			Author:    "user",
			Timestamp: time.Now().Add(time.Duration(i) * time.Second),
		}
		events[i].Content = genai.NewContentFromText(fmt.Sprintf("Message %d", i), genai.RoleUser)
	}
	return events
}

// CreateEventsWithContent creates n session events with the same content.
func CreateEventsWithContent(n int, content string) []*session.Event {
	events := make([]*session.Event, n)
	for i := 0; i < n; i++ {
		events[i] = &session.Event{
			ID:     string(rune('a' + i)),
			Author: "user",
		}
		events[i].Content = genai.NewContentFromText(content, genai.RoleUser)
	}
	return events
}

// FakeEmbedding generates a deterministic 1536-dim float32 vector from text.
// Uses SHA-256 hash of text to seed the vector values for reproducible tests.
func FakeEmbedding(text string) []float32 {
	hash := sha256.Sum256([]byte(text))
	vec := make([]float32, 1536)

	// Use hash bytes to generate vector components
	for i := 0; i < 1536; i++ {
		// Get 4 bytes for each float32
		idx := (i * 4) % len(hash)
		val := binary.BigEndian.Uint32(hash[idx:])
		// Normalize to [-1, 1] range
		vec[i] = (float32(val)/float32(math.MaxUint32))*2 - 1
	}

	return vec
}

// CosineSimilarity calculates cosine similarity between two vectors for test assertions.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
