// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package compaction

import (
	"context"
	"fmt"
	"time"

	"github.com/ieshan/adk-go-memory/adapter"
	"github.com/ieshan/idx"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// BackgroundCompactor provides APIs for external background compaction jobs.
// It operates on observations directly, separate from the session-based plugin.
//
// Example:
//
//	compactor := compaction.NewBackgroundCompactor(storage, llm)
//	obs, _ := compactor.QueryObservationsForCompaction(ctx, opts)
//	summary, _ := compactor.CreateCompactionSummary(ctx, obs, opts)
type BackgroundCompactor struct {
	storage adapter.Storage
	llm     model.LLM
}

// NewBackgroundCompactor creates a background compactor.
func NewBackgroundCompactor(storage adapter.Storage, llm model.LLM) *BackgroundCompactor {
	return &BackgroundCompactor{storage: storage, llm: llm}
}

// CompactionObservation wraps an observation with compaction metadata.
type CompactionObservation struct {
	adapter.Observation
	Age time.Duration
}

// CompactionQueryOptions controls QueryObservationsForCompaction.
type CompactionQueryOptions struct {
	OlderThan  time.Time
	MinAge     time.Duration
	MaxResults int
}

// QueryObservationsForCompaction returns observations for background compaction.
func (bc *BackgroundCompactor) QueryObservationsForCompaction(ctx context.Context, opts CompactionQueryOptions) ([]CompactionObservation, error) {
	searchOpts := &adapter.SearchOptions{Mode: adapter.SearchModeHybrid, MaxResults: opts.MaxResults}
	results, err := bc.storage.Search(ctx, searchOpts)
	if err != nil {
		return nil, fmt.Errorf("compaction: query observations: search failed: %w", err)
	}
	var candidates []CompactionObservation
	for _, result := range results {
		obs := result.Observation
		if !opts.OlderThan.IsZero() && obs.CreatedAt.After(opts.OlderThan) {
			continue
		}
		if opts.MinAge > 0 {
			if age := time.Since(obs.CreatedAt); age < opts.MinAge {
				continue
			}
		}
		candidates = append(candidates, CompactionObservation{Observation: obs, Age: time.Since(obs.CreatedAt)})
	}
	return candidates, nil
}

// SummaryOptions controls CreateCompactionSummary.
type SummaryOptions struct {
	Instruction string
	MaxTokens   int
	Tags        []string
}

const defaultBackgroundSummaryPrompt = `Consolidate these observations into a comprehensive summary capturing key facts, patterns, and relationships:
%s
Summary:`

// CreateCompactionSummary creates a summary observation from a batch.
func (bc *BackgroundCompactor) CreateCompactionSummary(ctx context.Context, observations []CompactionObservation, opts SummaryOptions) (*adapter.Observation, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("compaction: create summary: context cancelled: %w", err)
	}

	if bc.llm == nil {
		return nil, fmt.Errorf("compaction: create summary: LLM required")
	}
	if len(observations) == 0 {
		return nil, fmt.Errorf("compaction: create summary: no observations")
	}
	var text string
	for i, obs := range observations {
		text += fmt.Sprintf("\n[%d] %s: %s\n", i+1, obs.Level, obs.Content)
	}
	instruction := opts.Instruction
	if instruction == "" {
		instruction = defaultBackgroundSummaryPrompt
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText(fmt.Sprintf(instruction, text), genai.RoleUser)},
	}

	var resp *model.LLMResponse
	for r, err := range bc.llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return nil, fmt.Errorf("compaction: create summary: LLM failed: %w", err)
		}
		resp = r
		break // Non-streaming, take first response
	}

	summaryText := ExtractSummaryText(resp)
	if summaryText == "" {
		return nil, fmt.Errorf("compaction: create summary: empty summary")
	}
	base := observations[0].Observation
	return &adapter.Observation{
		ID: idx.NewID(), AppName: base.AppName, UserID: base.UserID,
		SessionID: base.SessionID, Content: summaryText,
		Level: adapter.LevelInductive, Tags: append([]string{"compaction_summary"}, opts.Tags...),
		CreatedAt: time.Now(),
	}, nil
}

// ArchiveObservations archives observations by tagging them.
func (bc *BackgroundCompactor) ArchiveObservations(ctx context.Context, observationIDs []idx.ID) error {
	for _, id := range observationIDs {
		obs, err := bc.storage.GetByID(ctx, id)
		if err != nil {
			return fmt.Errorf("compaction: archive: get %s: %w", id.String(), err)
		}
		obs.Tags = append(obs.Tags, "archived")
		// Delete and re-store to handle storage backends that don't support updates
		if err := bc.storage.Forget(ctx, id); err != nil {
			return fmt.Errorf("compaction: archive: forget %s: %w", id.String(), err)
		}
		if err := bc.storage.Store(ctx, obs); err != nil {
			return fmt.Errorf("compaction: archive: store %s: %w", id.String(), err)
		}
	}
	return nil
}

// PurgeArchivedObservations removes observations tagged as archived older than the cutoff.
func (bc *BackgroundCompactor) PurgeArchivedObservations(ctx context.Context, olderThan time.Time) (int, error) {
	// Note: adapter.Storage doesn't support tag-based filtering directly.
	// We retrieve all and filter manually, or use Purge with appropriate filters.
	results, err := bc.storage.Search(ctx, &adapter.SearchOptions{
		Mode: adapter.SearchModeHybrid,
	})
	if err != nil {
		return 0, fmt.Errorf("compaction: purge: search: %w", err)
	}
	var count int
	for _, result := range results {
		// Only purge if observation has "archived" tag and is older than cutoff
		if result.Observation.CreatedAt.Before(olderThan) && hasTag(result.Observation.Tags, "archived") {
			if err := bc.storage.Forget(ctx, result.Observation.ID); err != nil {
				return count, fmt.Errorf("compaction: purge: forget %s: %w", result.Observation.ID, err)
			}
			count++
		}
	}
	return count, nil
}

// hasTag checks if a slice contains a specific tag.
func hasTag(tags []string, target string) bool {
	for _, tag := range tags {
		if tag == target {
			return true
		}
	}
	return false
}
