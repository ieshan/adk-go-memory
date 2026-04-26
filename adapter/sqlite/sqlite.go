// Package sqlite provides SQLite storage implementation for the memory layer.
package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	sqlitevec "github.com/asg017/sqlite-vec-go-bindings/cgo"
	"github.com/ieshan/adk-go-memory/adapter"
	_ "github.com/mattn/go-sqlite3"
)

func init() {
	// Enable sqlite-vec for all SQLite connections
	sqlitevec.Auto()
}

// Compile-time interface compliance check.
var _ adapter.Storage = (*SQLiteStorage)(nil)

// SQLiteStorage implements Storage using SQLite with sqlite-vec and FTS5.
type SQLiteStorage struct {
	db *sql.DB
}

// InMemory creates a new in-memory SQLite storage instance.
func InMemory() (*SQLiteStorage, error) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		return nil, fmt.Errorf("sqlite: open memory db: %w", err)
	}

	storage := &SQLiteStorage{db: db}
	if err := storage.migrate(); err != nil {
		db.Close()
		return nil, err
	}

	return storage, nil
}

// NewSQLiteStorage creates a new file-based SQLite storage.
func NewSQLiteStorage(path string) (*SQLiteStorage, error) {
	db, err := sql.Open("sqlite3", path)
	if err != nil {
		return nil, fmt.Errorf("sqlite: open db: %w", err)
	}

	storage := &SQLiteStorage{db: db}
	if err := storage.migrate(); err != nil {
		db.Close()
		return nil, err
	}

	return storage, nil
}

// migrate creates the database schema.
func (s *SQLiteStorage) migrate() error {
	schema := `
CREATE TABLE IF NOT EXISTS observations (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    level TEXT NOT NULL,
    session_id TEXT NOT NULL,
    user_id TEXT,
    app_name TEXT,
    tags TEXT, -- JSON array
    times_derived INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    embedding BLOB -- JSON array of floats
);

CREATE INDEX IF NOT EXISTS idx_observations_session ON observations(session_id);
CREATE INDEX IF NOT EXISTS idx_observations_user ON observations(user_id);
CREATE INDEX IF NOT EXISTS idx_observations_app ON observations(app_name);
CREATE INDEX IF NOT EXISTS idx_observations_times_derived ON observations(times_derived DESC);
CREATE INDEX IF NOT EXISTS idx_observations_created_at ON observations(created_at DESC);
`
	if _, err := s.db.Exec(schema); err != nil {
		return fmt.Errorf("sqlite: migrate: %w", err)
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
	if _, err := s.db.Exec(vecSchema); err != nil {
		return fmt.Errorf("sqlite: create vec0 table: %w", err)
	}
	if _, err := s.db.Exec(ftsSchema); err != nil {
		return fmt.Errorf("sqlite: create fts5 table: %w", err)
	}

	return nil
}

// Store saves an observation to storage.
// The insert into the main table, FTS5, and vec0 virtual tables is wrapped
// in a transaction so that a partial failure does not leave the database
// in an inconsistent state.
func (s *SQLiteStorage) Store(ctx context.Context, obs *adapter.Observation) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite: store: begin tx: %w", err)
	}
	defer tx.Rollback()

	tagsJSON, _ := json.Marshal(obs.Tags)

	// Serialize embedding as JSON blob (for backup/retrieval)
	var embeddingBlob []byte
	if len(obs.Embedding) > 0 {
		embeddingBlob, _ = json.Marshal(obs.Embedding)
	}

	// Insert into main observations table
	result, err := tx.ExecContext(ctx,
		`INSERT INTO observations (id, content, level, session_id, user_id, app_name, tags, times_derived, created_at, embedding)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		obs.ID, obs.Content, obs.Level, obs.SessionID, obs.UserID, obs.AppName,
		string(tagsJSON), obs.TimesDerived, obs.CreatedAt, embeddingBlob)
	if err != nil {
		return fmt.Errorf("sqlite: store: %w", err)
	}

	// Get the rowid for virtual table inserts
	rowID, err := result.LastInsertId()
	if err != nil {
		return fmt.Errorf("sqlite: get last insert id: %w", err)
	}

	// Insert into FTS5 table for text search
	if _, err := tx.ExecContext(ctx,
		`INSERT INTO observations_fts(rowid, content) VALUES (?, ?)`,
		rowID, obs.Content); err != nil {
		return fmt.Errorf("sqlite: store fts5: %w", err)
	}

	// Insert into vec0 table for vector search
	if len(obs.Embedding) > 0 {
		embeddingSerialized, err := sqlitevec.SerializeFloat32(obs.Embedding)
		if err != nil {
			return fmt.Errorf("sqlite: serialize embedding: %w", err)
		}
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO vec_observations(rowid, embedding) VALUES (?, ?)`,
			rowID, embeddingSerialized); err != nil {
			return fmt.Errorf("sqlite: store vec0: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("sqlite: store: commit: %w", err)
	}

	return nil
}

// GetByID retrieves an observation by its ID.
func (s *SQLiteStorage) GetByID(ctx context.Context, id string) (*adapter.Observation, error) {
	var obs adapter.Observation
	var tagsJSON string
	var embeddingBlob []byte

	err := s.db.QueryRowContext(ctx,
		`SELECT id, content, level, session_id, user_id, app_name, tags, times_derived, created_at, embedding
		 FROM observations WHERE id = ?`, id).Scan(
		&obs.ID, &obs.Content, &obs.Level, &obs.SessionID,
		&obs.UserID, &obs.AppName, &tagsJSON, &obs.TimesDerived,
		&obs.CreatedAt, &embeddingBlob)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("observation not found: %s", id)
		}
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

	return &obs, nil
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
	rows, err := s.db.QueryContext(ctx,
		`SELECT o.id, o.content, o.level, o.session_id, o.user_id, o.app_name, 
		        o.tags, o.times_derived, o.created_at, o.embedding, v.distance
		 FROM vec_observations v
		 JOIN observations o ON o.rowid = v.rowid
		 WHERE v.embedding MATCH ? AND k = ?`+filterClause+`
		 ORDER BY v.distance
		 LIMIT ?`, args...)
	if err != nil {
		return nil, fmt.Errorf("sqlite: vector search: %w", err)
	}
	defer rows.Close()

	return s.scanResultsWithDistance(rows, "vector")
}

// queryRecentAsSearchResults returns recent observations when no embedding provided
func (s *SQLiteStorage) queryRecentAsSearchResults(ctx context.Context, opts *adapter.SearchOptions) ([]adapter.SearchResult, error) {
	whereClause, args := s.buildWhereClause(opts)
	query := fmt.Sprintf(
		`SELECT id, content, level, session_id, user_id, app_name, tags, times_derived, created_at, embedding
		 FROM observations
		 WHERE %s
		 ORDER BY created_at DESC
		 LIMIT ?`, whereClause)
	args = append(args, opts.MaxResults)

	rows, err := s.db.QueryContext(ctx, query, args...)
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
	rows, err := s.db.QueryContext(ctx,
		`SELECT o.id, o.content, o.level, o.session_id, o.user_id, o.app_name,
		        o.tags, o.times_derived, o.created_at, o.embedding, bm25(observations_fts)
		 FROM observations_fts f
		 JOIN observations o ON o.rowid = f.rowid
		 WHERE f.observations_fts MATCH ?`+filterClause+`
		 ORDER BY bm25(observations_fts)
		 LIMIT ?`, args...)
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
	vectorRanks := make(map[string]int)
	for i, r := range vectorResults {
		vectorRanks[r.Observation.ID] = i + 1
	}

	ftsRanks := make(map[string]int)
	for i, r := range ftsResults {
		ftsRanks[r.Observation.ID] = i + 1
	}

	// Collect all unique IDs
	allIDs := make(map[string]bool)
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

// buildWhereClause creates WHERE clause and args for filtered queries
func (s *SQLiteStorage) buildWhereClause(opts *adapter.SearchOptions) (string, []interface{}) {
	conditions := []string{"1=1"}
	args := []interface{}{}

	if opts.SessionID != "" {
		conditions = append(conditions, "session_id = ?")
		args = append(args, opts.SessionID)
	}
	if opts.UserID != "" {
		conditions = append(conditions, "user_id = ?")
		args = append(args, opts.UserID)
	}
	if opts.AppName != "" {
		conditions = append(conditions, "app_name = ?")
		args = append(args, opts.AppName)
	}

	return strings.Join(conditions, " AND "), args
}

// Forget deletes an observation by ID.
// All deletes (main table, FTS5, vec0) are wrapped in a transaction
// so that a partial failure does not leave the database in an inconsistent state.
func (s *SQLiteStorage) Forget(ctx context.Context, id string) error {
	// Get rowid first for virtual table cleanup
	var rowid int64
	err := s.db.QueryRowContext(ctx, `SELECT rowid FROM observations WHERE id = ?`, id).Scan(&rowid)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil // Already deleted
		}
		return err
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite: forget: begin tx: %w", err)
	}
	defer tx.Rollback()

	// Delete from main table
	if _, err := tx.ExecContext(ctx, `DELETE FROM observations WHERE id = ?`, id); err != nil {
		return fmt.Errorf("sqlite: forget main: %w", err)
	}

	// Delete from FTS5
	if _, err := tx.ExecContext(ctx, `DELETE FROM observations_fts WHERE rowid = ?`, rowid); err != nil {
		return fmt.Errorf("sqlite: forget fts5: %w", err)
	}

	// Delete from vec0
	if _, err := tx.ExecContext(ctx, `DELETE FROM vec_observations WHERE rowid = ?`, rowid); err != nil {
		return fmt.Errorf("sqlite: forget vec0: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("sqlite: forget: commit: %w", err)
	}

	return nil
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
	whereClause := "1=1"
	args := []interface{}{}

	if sessionID, ok := filter["session_id"]; ok {
		whereClause += ` AND session_id = ?`
		args = append(args, sessionID)
	}
	if userID, ok := filter["user_id"]; ok {
		whereClause += ` AND user_id = ?`
		args = append(args, userID)
	}
	if appName, ok := filter["app_name"]; ok {
		whereClause += ` AND app_name = ?`
		args = append(args, appName)
	}

	// Require at least one recognized filter to prevent accidental full deletion
	if len(args) == 0 {
		return fmt.Errorf("sqlite: purge: at least one filter key (session_id, user_id, app_name) is required")
	}

	// Get rowids for virtual table cleanup (before transaction, since this is a read)
	rowidQuery := fmt.Sprintf(`SELECT rowid FROM observations WHERE %s`, whereClause)
	rows, err := s.db.QueryContext(ctx, rowidQuery, args...)
	if err != nil {
		return err
	}
	defer rows.Close()

	var rowids []int64
	for rows.Next() {
		var rowid int64
		if err := rows.Scan(&rowid); err != nil {
			return err
		}
		rowids = append(rowids, rowid)
	}
	if err := rows.Err(); err != nil {
		return err
	}

	if len(rowids) == 0 {
		return nil // Nothing to delete
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite: purge: begin tx: %w", err)
	}
	defer tx.Rollback()

	// Delete from main table
	deleteQuery := fmt.Sprintf(`DELETE FROM observations WHERE %s`, whereClause)
	_, err = tx.ExecContext(ctx, deleteQuery, args...)
	if err != nil {
		return fmt.Errorf("sqlite: purge main: %w", err)
	}

	// Clean up virtual tables
	for _, rowid := range rowids {
		if _, err := tx.ExecContext(ctx, `DELETE FROM observations_fts WHERE rowid = ?`, rowid); err != nil {
			return fmt.Errorf("sqlite: purge fts5: %w", err)
		}
		if _, err := tx.ExecContext(ctx, `DELETE FROM vec_observations WHERE rowid = ?`, rowid); err != nil {
			return fmt.Errorf("sqlite: purge vec0: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("sqlite: purge: commit: %w", err)
	}

	return nil
}

// IncrementTimesDerived increments the times_derived counter for an observation.
// Returns an error if the observation does not exist.
func (s *SQLiteStorage) IncrementTimesDerived(ctx context.Context, id string) error {
	result, err := s.db.ExecContext(ctx,
		`UPDATE observations SET times_derived = times_derived + 1 WHERE id = ?`, id)
	if err != nil {
		return err
	}
	n, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return fmt.Errorf("sqlite: increment times_derived: observation not found: %s", id)
	}
	return nil
}

// Close releases the database connection.
func (s *SQLiteStorage) Close() error {
	return s.db.Close()
}

// QueryMostDerived returns observations sorted by times_derived DESC.
// Most-derived facts (referenced multiple times) are returned first.
func (s *SQLiteStorage) QueryMostDerived(ctx context.Context, sessionID, userID, appName string, limit int) ([]adapter.Observation, error) {
	whereClause, args := s.buildWhereClause(&adapter.SearchOptions{
		SessionID: sessionID,
		UserID:    userID,
		AppName:   appName,
	})

	query := fmt.Sprintf(
		`SELECT id, content, level, session_id, user_id, app_name, tags, times_derived, created_at, embedding
		 FROM observations
		 WHERE %s
		 ORDER BY times_derived DESC, created_at DESC
		 LIMIT ?`, whereClause)
	args = append(args, limit)

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("sqlite: query most derived: %w", err)
	}
	defer rows.Close()

	return s.scanObservations(rows)
}

// QueryRecent returns observations sorted by created_at DESC.
// Most recent observations are returned first.
func (s *SQLiteStorage) QueryRecent(ctx context.Context, sessionID, userID, appName string, limit int) ([]adapter.Observation, error) {
	whereClause, args := s.buildWhereClause(&adapter.SearchOptions{
		SessionID: sessionID,
		UserID:    userID,
		AppName:   appName,
	})

	query := fmt.Sprintf(
		`SELECT id, content, level, session_id, user_id, app_name, tags, times_derived, created_at, embedding
		 FROM observations
		 WHERE %s
		 ORDER BY created_at DESC
		 LIMIT ?`, whereClause)
	args = append(args, limit)

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("sqlite: query recent: %w", err)
	}
	defer rows.Close()

	return s.scanObservations(rows)
}

// scanObservations scans rows into Observation structs (without SearchResult wrapper)
func (s *SQLiteStorage) scanObservations(rows *sql.Rows) ([]adapter.Observation, error) {
	var observations []adapter.Observation
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

		observations = append(observations, obs)
	}

	return observations, rows.Err()
}
