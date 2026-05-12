// Package sqlite provides SQLite storage implementation for the memory layer.
package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"

	sqlitevec "github.com/asg017/sqlite-vec-go-bindings/cgo"
	"github.com/ieshan/adk-go-memory/adapter"
	"github.com/ieshan/idx"
	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
)

func init() {
	// Enable sqlite-vec for all SQLite connections
	sqlitevec.Auto()
}

// Compile-time interface compliance check.
var _ adapter.Storage = (*SQLiteStorage)(nil)

// SQLiteStorage implements Storage using SQLite with sqlite-vec and FTS5.
type SQLiteStorage struct {
	db    *gorm.DB
	ownDB bool // true if we opened the connection and should close it
}

// InMemory creates a new in-memory SQLite storage instance.
func InMemory() (*SQLiteStorage, error) {
	db, err := gorm.Open(sqlite.Open("file::memory:?cache=shared"), &gorm.Config{})
	if err != nil {
		return nil, fmt.Errorf("sqlite: open memory db: %w", err)
	}

	storage := &SQLiteStorage{db: db, ownDB: true}
	if err := storage.migrate(); err != nil {
		return nil, err
	}

	return storage, nil
}

// NewSQLiteStorage creates a new file-based SQLite storage.
func NewSQLiteStorage(path string) (*SQLiteStorage, error) {
	db, err := gorm.Open(sqlite.Open(path), &gorm.Config{})
	if err != nil {
		return nil, fmt.Errorf("sqlite: open db: %w", err)
	}

	storage := &SQLiteStorage{db: db, ownDB: true}
	if err := storage.migrate(); err != nil {
		return nil, err
	}

	return storage, nil
}

// NewSQLiteStorageWithGORM creates a new SQLiteStorage from an existing GORM connection.
// The caller retains ownership of the provided *gorm.DB and is responsible for closing it.
// This is useful when integrating with existing connection pools or when the database
// connection needs to be shared across multiple components.
func NewSQLiteStorageWithGORM(db *gorm.DB) (*SQLiteStorage, error) {
	if db == nil {
		return nil, fmt.Errorf("sqlite: NewSQLiteStorageWithGORM: db is nil")
	}
	storage := &SQLiteStorage{db: db, ownDB: false}
	if err := storage.migrate(); err != nil {
		return nil, err
	}
	return storage, nil
}

// migrate creates the database schema.
func (s *SQLiteStorage) migrate() error {
	// AutoMigrate creates the main table and indexes from the GORM model.
	if err := s.db.AutoMigrate(&StorageObservation{}); err != nil {
		return fmt.Errorf("sqlite: auto migrate: %w", err)
	}

	// sqlite-vec virtual table for vector similarity search
	vecSchema := `
CREATE VIRTUAL TABLE IF NOT EXISTS vec_observations USING vec0(
    embedding float[1536]
)
`

	// FTS5 virtual table for full-text search
	ftsSchema := `
CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
    content,
    content_rowid=rowid
)
`

	// Execute all schema statements
	if err := s.db.Exec(vecSchema).Error; err != nil {
		return fmt.Errorf("sqlite: create vec0 table: %w", err)
	}
	if err := s.db.Exec(ftsSchema).Error; err != nil {
		return fmt.Errorf("sqlite: create fts5 table: %w", err)
	}

	return nil
}

// toStorageObservation maps an adapter.Observation to a StorageObservation.
func toStorageObservation(obs *adapter.Observation) *StorageObservation {
	tagsJSON, _ := json.Marshal(obs.Tags)
	var embeddingBlob []byte
	if len(obs.Embedding) > 0 {
		embeddingBlob, _ = json.Marshal(obs.Embedding)
	}
	return &StorageObservation{
		ID:           obs.ID,
		Content:      obs.Content,
		Level:        string(obs.Level),
		SessionID:    obs.SessionID,
		UserID:       obs.UserID,
		AppName:      obs.AppName,
		Tags:         string(tagsJSON),
		TimesDerived: obs.TimesDerived,
		CreatedAt:    obs.CreatedAt,
		Embedding:    embeddingBlob,
	}
}

// toAdapterObservation maps a StorageObservation to an adapter.Observation.
func toAdapterObservation(sobs *StorageObservation) (*adapter.Observation, error) {
	obs := &adapter.Observation{
		ID:           sobs.ID,
		Content:      sobs.Content,
		Level:        adapter.ObservationLevel(sobs.Level),
		SessionID:    sobs.SessionID,
		UserID:       sobs.UserID,
		AppName:      sobs.AppName,
		TimesDerived: sobs.TimesDerived,
		CreatedAt:    sobs.CreatedAt,
	}
	if sobs.Tags != "" {
		if err := json.Unmarshal([]byte(sobs.Tags), &obs.Tags); err != nil {
			return nil, fmt.Errorf("sqlite: unmarshal tags: %w", err)
		}
	}
	if len(sobs.Embedding) > 0 {
		if err := json.Unmarshal(sobs.Embedding, &obs.Embedding); err != nil {
			return nil, fmt.Errorf("sqlite: unmarshal embedding: %w", err)
		}
	}
	return obs, nil
}

// scopedQuery returns a GORM query scoped by session_id, user_id, and app_name.
func (s *SQLiteStorage) scopedQuery(db *gorm.DB, sessionID, userID, appName string) *gorm.DB {
	if sessionID != "" {
		db = db.Where("session_id = ?", sessionID)
	}
	if userID != "" {
		db = db.Where("user_id = ?", userID)
	}
	if appName != "" {
		db = db.Where("app_name = ?", appName)
	}
	return db
}

// Store saves an observation to storage.
// The insert into the main table, FTS5, and vec0 virtual tables is wrapped
// in a transaction so that a partial failure does not leave the database
// in an inconsistent state.
func (s *SQLiteStorage) Store(ctx context.Context, obs *adapter.Observation) error {
	sobs := toStorageObservation(obs)

	return s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		// Insert into main observations table
		if err := tx.Create(sobs).Error; err != nil {
			return fmt.Errorf("sqlite: store: %w", err)
		}

		rowID := sobs.RowID

		// Insert into FTS5 table for text search
		if err := tx.Exec(
			`INSERT INTO observations_fts(rowid, content) VALUES (?, ?)`,
			rowID, obs.Content).Error; err != nil {
			return fmt.Errorf("sqlite: store fts5: %w", err)
		}

		// Insert into vec0 table for vector search
		if len(obs.Embedding) > 0 {
			embeddingSerialized, err := sqlitevec.SerializeFloat32(obs.Embedding)
			if err != nil {
				return fmt.Errorf("sqlite: serialize embedding: %w", err)
			}
			if err := tx.Exec(
				`INSERT INTO vec_observations(rowid, embedding) VALUES (?, ?)`,
				rowID, embeddingSerialized).Error; err != nil {
				return fmt.Errorf("sqlite: store vec0: %w", err)
			}
		}

		return nil
	})
}

// GetByID retrieves an observation by its ID.
func (s *SQLiteStorage) GetByID(ctx context.Context, id idx.ID) (*adapter.Observation, error) {
	var sobs StorageObservation
	if err := s.db.WithContext(ctx).First(&sobs, "id = ?", id).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, fmt.Errorf("observation not found: %s", id.String())
		}
		return nil, err
	}
	return toAdapterObservation(&sobs)
}

// Search finds observations matching the given options.
func (s *SQLiteStorage) Search(ctx context.Context, opts *adapter.SearchOptions) ([]adapter.SearchResult, error) {
	maxResults := opts.MaxResults
	if maxResults == 0 {
		maxResults = 10
	}

	// Use a copy to avoid mutating the caller's struct.
	o := *opts
	o.MaxResults = maxResults

	switch o.Mode {
	case adapter.SearchModeVector:
		return s.searchVector(ctx, &o)
	case adapter.SearchModeFTS:
		return s.searchFTS(ctx, &o)
	case adapter.SearchModeHybrid:
		return s.searchHybrid(ctx, &o)
	default:
		return s.searchHybrid(ctx, &o)
	}
}

func (s *SQLiteStorage) searchVector(ctx context.Context, opts *adapter.SearchOptions) ([]adapter.SearchResult, error) {
	if len(opts.Embedding) == 0 {
		// No embedding provided - fall back to recent observations
		return s.queryRecentAsSearchResults(ctx, opts)
	}

	embeddingSerialized, err := sqlitevec.SerializeFloat32(opts.Embedding)
	if err != nil {
		return nil, fmt.Errorf("sqlite: serialize query embedding: %w", err)
	}

	// Build filter clause
	var filterClause string
	var filterArgs []interface{}

	if opts.SessionID != "" {
		filterClause += " AND o.session_id = ?"
		filterArgs = append(filterArgs, opts.SessionID)
	}
	if opts.UserID != "" {
		filterClause += " AND o.user_id = ?"
		filterArgs = append(filterArgs, opts.UserID)
	}
	if opts.AppName != "" {
		filterClause += " AND o.app_name = ?"
		filterArgs = append(filterArgs, opts.AppName)
	}

	// Query vec_observations for similar vectors
	// Join back to observations to get full data
	// Parameters: embedding, k, [filters...], limit
	args := append([]interface{}{embeddingSerialized, opts.MaxResults}, filterArgs...)
	args = append(args, opts.MaxResults)
	rows, err := s.db.WithContext(ctx).Raw(
		`SELECT o.id, o.content, o.level, o.session_id, o.user_id, o.app_name, 
		        o.tags, o.times_derived, o.created_at, o.embedding, v.distance
		 FROM vec_observations v
		 JOIN observations o ON o.rowid = v.rowid
		 WHERE v.embedding MATCH ? AND k = ?`+filterClause+`
		 ORDER BY v.distance
		 LIMIT ?`, args...).Rows()
	if err != nil {
		return nil, fmt.Errorf("sqlite: vector search: %w", err)
	}
	defer rows.Close()

	return s.scanResultsWithDistance(rows, "vector")
}

// queryRecentAsSearchResults returns recent observations when no embedding provided
func (s *SQLiteStorage) queryRecentAsSearchResults(ctx context.Context, opts *adapter.SearchOptions) ([]adapter.SearchResult, error) {
	query := s.scopedQuery(s.db.WithContext(ctx), opts.SessionID, opts.UserID, opts.AppName)
	query = query.Order("id DESC").Limit(opts.MaxResults)

	rows, err := query.Model(&StorageObservation{}).Select(
		"id", "content", "level", "session_id", "user_id", "app_name",
		"tags", "times_derived", "created_at", "embedding",
	).Rows()
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	results, err := s.scanResults(rows, "vector_fallback")
	if err != nil {
		return nil, err
	}

	// Assign decaying scores for fallback
	for i := range results {
		results[i].Score = 1.0 - float64(i)*0.1
		if results[i].Score < 0.1 {
			results[i].Score = 0.1
		}
	}
	return results, nil
}

// sanitizeFTS5Query extracts clean tokens from a query for safe FTS5 matching.
// It removes punctuation and special characters that could cause syntax errors.
// Tokens are lowercased to neutralize FTS5 operators (AND, OR, NOT, NEAR)
// which are only recognized in uppercase.
func sanitizeFTS5Query(query string) string {
	var tokens []string
	var current strings.Builder

	for _, r := range query {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			// Lowercase to neutralize FTS5 operators (AND, OR, NOT, NEAR)
			if r >= 'A' && r <= 'Z' {
				r += 32
			}
			current.WriteRune(r)
		} else if current.Len() > 0 {
			tokens = append(tokens, current.String())
			current.Reset()
		}
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}

	return strings.Join(tokens, " ")
}

func (s *SQLiteStorage) searchFTS(ctx context.Context, opts *adapter.SearchOptions) ([]adapter.SearchResult, error) {
	if opts.Query == "" {
		// No query - return recent observations
		return s.queryRecentAsSearchResults(ctx, opts)
	}

	// Build WHERE clause for filtering
	var filterClause string
	var filterArgs []interface{}

	if opts.SessionID != "" {
		filterClause += " AND o.session_id = ?"
		filterArgs = append(filterArgs, opts.SessionID)
	}
	if opts.UserID != "" {
		filterClause += " AND o.user_id = ?"
		filterArgs = append(filterArgs, opts.UserID)
	}
	if opts.AppName != "" {
		filterClause += " AND o.app_name = ?"
		filterArgs = append(filterArgs, opts.AppName)
	}

	// Sanitize query for FTS5 MATCH syntax
	// Extract only alphanumeric tokens and join with spaces for safe FTS5 matching
	escapedQuery := sanitizeFTS5Query(opts.Query)

	// Use FTS5 MATCH with bm25 ranking
	// bm25 returns lower values for better matches
	args := append([]interface{}{escapedQuery}, append(filterArgs, opts.MaxResults)...)
	rows, err := s.db.WithContext(ctx).Raw(
		`SELECT o.id, o.content, o.level, o.session_id, o.user_id, o.app_name,
		        o.tags, o.times_derived, o.created_at, o.embedding, bm25(observations_fts)
		 FROM observations_fts f
		 JOIN observations o ON o.rowid = f.rowid
		 WHERE f.observations_fts MATCH ?`+filterClause+`
		 ORDER BY bm25(observations_fts)
		 LIMIT ?`, args...).Rows()
	if err != nil {
		// If FTS5 syntax error, fall back to empty results rather than failing
		if strings.Contains(err.Error(), "fts5: syntax error") {
			return s.queryRecentAsSearchResults(ctx, opts)
		}
		return nil, fmt.Errorf("sqlite: fts search: %w", err)
	}
	defer rows.Close()

	return s.scanResultsWithDistance(rows, "fts")
}

func (s *SQLiteStorage) searchHybrid(ctx context.Context, opts *adapter.SearchOptions) ([]adapter.SearchResult, error) {
	// Get results from both methods
	vectorOpts := *opts
	vectorOpts.Mode = adapter.SearchModeVector
	vectorOpts.MaxResults = opts.MaxResults * 2

	ftsOpts := *opts
	ftsOpts.Mode = adapter.SearchModeFTS
	ftsOpts.MaxResults = opts.MaxResults * 2

	vectorResults, err := s.searchVector(ctx, &vectorOpts)
	if err != nil {
		return nil, err
	}

	ftsResults, err := s.searchFTS(ctx, &ftsOpts)
	if err != nil {
		return nil, err
	}

	// Reciprocal Rank Fusion (RRF)
	// score = sum(1/(k + rank)) for each list where item appears
	const k = 60

	// Build rank maps
	vectorRanks := make(map[idx.ID]int)
	for i, r := range vectorResults {
		vectorRanks[r.Observation.ID] = i + 1
	}

	ftsRanks := make(map[idx.ID]int)
	for i, r := range ftsResults {
		ftsRanks[r.Observation.ID] = i + 1
	}

	// Collect all unique IDs
	allIDs := make(map[idx.ID]bool)
	for id := range vectorRanks {
		allIDs[id] = true
	}
	for id := range ftsRanks {
		allIDs[id] = true
	}

	// Calculate RRF scores
	type rrfItem struct {
		obs   adapter.Observation
		score float64
	}
	rrfItems := make([]rrfItem, 0, len(allIDs))

	for id := range allIDs {
		var obs *adapter.Observation
		for _, r := range vectorResults {
			if r.Observation.ID == id {
				obs = &r.Observation
				break
			}
		}
		if obs == nil {
			for _, r := range ftsResults {
				if r.Observation.ID == id {
					obs = &r.Observation
					break
				}
			}
		}
		if obs == nil {
			continue
		}

		score := 0.0
		if rank, ok := vectorRanks[id]; ok {
			score += 1.0 / (k + float64(rank))
		}
		if rank, ok := ftsRanks[id]; ok {
			score += 1.0 / (k + float64(rank))
		}

		rrfItems = append(rrfItems, rrfItem{obs: *obs, score: score})
	}

	// Sort by RRF score descending
	for i := 0; i < len(rrfItems); i++ {
		for j := i + 1; j < len(rrfItems); j++ {
			if rrfItems[j].score > rrfItems[i].score {
				rrfItems[i], rrfItems[j] = rrfItems[j], rrfItems[i]
			}
		}
	}

	// Take top results
	limit := opts.MaxResults
	if limit > len(rrfItems) {
		limit = len(rrfItems)
	}

	results := make([]adapter.SearchResult, limit)
	for i := 0; i < limit; i++ {
		results[i] = adapter.SearchResult{
			Observation: rrfItems[i].obs,
			Score:       rrfItems[i].score,
			Source:      "rrf",
		}
	}

	return results, nil
}

// scanResults scans rows into SearchResult structs.
func (s *SQLiteStorage) scanResults(rows *sql.Rows, source string) ([]adapter.SearchResult, error) {
	var results []adapter.SearchResult
	for rows.Next() {
		var obs adapter.Observation
		var tagsJSON string
		var embeddingBlob []byte

		err := rows.Scan(&obs.ID, &obs.Content, &obs.Level, &obs.SessionID,
			&obs.UserID, &obs.AppName, &tagsJSON, &obs.TimesDerived,
			&obs.CreatedAt, &embeddingBlob)
		if err != nil {
			return nil, err
		}

		if tagsJSON != "" {
			if err := json.Unmarshal([]byte(tagsJSON), &obs.Tags); err != nil {
				return nil, fmt.Errorf("sqlite: unmarshal tags: %w", err)
			}
		}
		if len(embeddingBlob) > 0 {
			if err := json.Unmarshal(embeddingBlob, &obs.Embedding); err != nil {
				return nil, fmt.Errorf("sqlite: unmarshal embedding: %w", err)
			}
		}

		results = append(results, adapter.SearchResult{
			Observation: obs,
			Score:       obs.Score(),
			Source:      source,
		})
	}

	return results, rows.Err()
}

// scanResultsWithDistance scans rows that include a distance/score column
func (s *SQLiteStorage) scanResultsWithDistance(rows *sql.Rows, source string) ([]adapter.SearchResult, error) {
	var results []adapter.SearchResult
	for rows.Next() {
		var obs adapter.Observation
		var tagsJSON string
		var embeddingBlob []byte
		var distance float64

		err := rows.Scan(&obs.ID, &obs.Content, &obs.Level, &obs.SessionID,
			&obs.UserID, &obs.AppName, &tagsJSON, &obs.TimesDerived,
			&obs.CreatedAt, &embeddingBlob, &distance)
		if err != nil {
			return nil, err
		}

		// Deserialize tags
		if tagsJSON != "" {
			if err := json.Unmarshal([]byte(tagsJSON), &obs.Tags); err != nil {
				return nil, fmt.Errorf("sqlite: unmarshal tags: %w", err)
			}
		}

		// Deserialize embedding
		if len(embeddingBlob) > 0 {
			if err := json.Unmarshal(embeddingBlob, &obs.Embedding); err != nil {
				return nil, fmt.Errorf("sqlite: unmarshal embedding: %w", err)
			}
		}

		// Convert distance to score (lower distance = higher score)
		// For vec0: distance is L2 distance, smaller is better
		// For bm25: lower is better, but values can be negative
		score := 1.0 / (1.0 + math.Abs(distance))

		results = append(results, adapter.SearchResult{
			Observation: obs,
			Score:       score,
			Source:      source,
		})
	}

	return results, rows.Err()
}

// Forget deletes an observation by ID.
// All deletes (main table, FTS5, vec0) are wrapped in a transaction
// so that a partial failure does not leave the database in an inconsistent state.
func (s *SQLiteStorage) Forget(ctx context.Context, id idx.ID) error {
	return s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		// Get rowid first for virtual table cleanup
		var sobs StorageObservation
		if err := tx.Select("rowid").First(&sobs, "id = ?", id).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return nil // Already deleted
			}
			return err
		}
		rowid := sobs.RowID

		// Delete from main table
		if err := tx.Delete(&StorageObservation{}, "id = ?", id).Error; err != nil {
			return fmt.Errorf("sqlite: forget main: %w", err)
		}

		// Delete from FTS5
		if err := tx.Exec(`DELETE FROM observations_fts WHERE rowid = ?`, rowid).Error; err != nil {
			return fmt.Errorf("sqlite: forget fts5: %w", err)
		}

		// Delete from vec0
		if err := tx.Exec(`DELETE FROM vec_observations WHERE rowid = ?`, rowid).Error; err != nil {
			return fmt.Errorf("sqlite: forget vec0: %w", err)
		}

		return nil
	})
}

// Purge deletes observations matching the filter.
// At least one recognized filter key (session_id, user_id, app_name) must be
// provided; otherwise an error is returned to prevent accidental full deletion.
// Unrecognized filter keys are rejected to prevent silent partial matching.
// All deletes (main table, FTS5, vec0) are wrapped in a transaction
// so that a partial failure does not leave the database in an inconsistent state.
func (s *SQLiteStorage) Purge(ctx context.Context, filter map[string]string) error {
	// Validate filter keys — reject any unrecognized keys
	validKeys := map[string]bool{"session_id": true, "user_id": true, "app_name": true}
	for key := range filter {
		if !validKeys[key] {
			return fmt.Errorf("sqlite: purge: unrecognized filter key %q (valid keys: session_id, user_id, app_name)", key)
		}
	}

	// Build WHERE clause
	sessionID := filter["session_id"]
	userID := filter["user_id"]
	appName := filter["app_name"]

	// Require at least one recognized filter to prevent accidental full deletion
	if sessionID == "" && userID == "" && appName == "" {
		return fmt.Errorf("sqlite: purge: at least one filter key (session_id, user_id, app_name) is required")
	}

	return s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		// Collect rowids for virtual table cleanup
		query := s.scopedQuery(tx, sessionID, userID, appName)
		var results []StorageObservation
		if err := query.Select("rowid").Find(&results).Error; err != nil {
			return err
		}

		if len(results) == 0 {
			return nil // Nothing to delete
		}

		// Delete from main table
		deleteQuery := s.scopedQuery(tx, sessionID, userID, appName)
		if err := deleteQuery.Delete(&StorageObservation{}).Error; err != nil {
			return fmt.Errorf("sqlite: purge main: %w", err)
		}

		// Clean up virtual tables
		for _, sobs := range results {
			if err := tx.Exec(`DELETE FROM observations_fts WHERE rowid = ?`, sobs.RowID).Error; err != nil {
				return fmt.Errorf("sqlite: purge fts5: %w", err)
			}
			if err := tx.Exec(`DELETE FROM vec_observations WHERE rowid = ?`, sobs.RowID).Error; err != nil {
				return fmt.Errorf("sqlite: purge vec0: %w", err)
			}
		}

		return nil
	})
}

// IncrementTimesDerived increments the times_derived counter for an observation.
// Returns an error if the observation does not exist.
func (s *SQLiteStorage) IncrementTimesDerived(ctx context.Context, id idx.ID) error {
	result := s.db.WithContext(ctx).Model(&StorageObservation{}).Where("id = ?", id).UpdateColumn("times_derived", gorm.Expr("times_derived + 1"))
	if result.Error != nil {
		return result.Error
	}
	if result.RowsAffected == 0 {
		return fmt.Errorf("sqlite: increment times_derived: observation not found: %s", id.String())
	}
	return nil
}

// Close releases the database connection.
// Note: When using NewSQLiteStorageWithGORM, the caller retains ownership
// of the *gorm.DB connection and Close() does not close it.
func (s *SQLiteStorage) Close() error {
	if s.ownDB {
		sqlDB, err := s.db.DB()
		if err != nil {
			return err
		}
		return sqlDB.Close()
	}
	return nil
}

// QueryMostDerived returns observations sorted by times_derived DESC.
// Most-derived facts (referenced multiple times) are returned first.
func (s *SQLiteStorage) QueryMostDerived(ctx context.Context, sessionID, userID, appName string, limit int) ([]adapter.Observation, error) {
	query := s.scopedQuery(s.db.WithContext(ctx), sessionID, userID, appName)
	var results []StorageObservation
	if err := query.Order("times_derived DESC, id DESC").Limit(limit).Find(&results).Error; err != nil {
		return nil, fmt.Errorf("sqlite: query most derived: %w", err)
	}

	observations := make([]adapter.Observation, 0, len(results))
	for i := range results {
		obs, err := toAdapterObservation(&results[i])
		if err != nil {
			return nil, err
		}
		observations = append(observations, *obs)
	}
	return observations, nil
}

// QueryRecent returns observations sorted by id DESC (ULID-based, time-ordered).
// Most recent observations are returned first.
func (s *SQLiteStorage) QueryRecent(ctx context.Context, sessionID, userID, appName string, limit int) ([]adapter.Observation, error) {
	query := s.scopedQuery(s.db.WithContext(ctx), sessionID, userID, appName)
	var results []StorageObservation
	if err := query.Order("id DESC").Limit(limit).Find(&results).Error; err != nil {
		return nil, fmt.Errorf("sqlite: query recent: %w", err)
	}

	observations := make([]adapter.Observation, 0, len(results))
	for i := range results {
		obs, err := toAdapterObservation(&results[i])
		if err != nil {
			return nil, err
		}
		observations = append(observations, *obs)
	}
	return observations, nil
}
