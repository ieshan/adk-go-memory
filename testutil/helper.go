// Package testutil provides test utilities for the adk-go-memory project.
package testutil

import (
	"crypto/sha256"
	"encoding/binary"
	"math"
)

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
