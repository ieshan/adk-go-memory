package sqlite

import (
	"time"

	"github.com/ieshan/idx"
)

// StorageObservation is the GORM model for the observations table.
// It mirrors the existing schema so that AutoMigrate can create tables and indexes.
type StorageObservation struct {
	RowID        int64     `gorm:"primaryKey;autoIncrement;column:rowid"`
	ID           idx.ID    `gorm:"uniqueIndex;not null;column:id"`
	Content      string    `gorm:"not null;column:content"`
	Level        string    `gorm:"not null;column:level"`
	SessionID    string    `gorm:"index:idx_observations_session;column:session_id"`
	UserID       string    `gorm:"index:idx_observations_user;column:user_id"`
	AppName      string    `gorm:"index:idx_observations_app;column:app_name"`
	Tags         string    `gorm:"column:tags"`
	TimesDerived int       `gorm:"default:1;column:times_derived"`
	CreatedAt    time.Time `gorm:"column:created_at"`
	Embedding    []byte    `gorm:"column:embedding"`
}

// TableName explicitly sets the table name for the StorageObservation struct.
func (StorageObservation) TableName() string {
	return "observations"
}
