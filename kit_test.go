package memory

import (
	"testing"

	"github.com/ieshan/adk-go-memory/adapter"
	"github.com/ieshan/adk-go-memory/compaction"
)

func TestNew_ValidConfig(t *testing.T) {
	storage := adapter.InMemory()
	defer storage.Close()

	kit, err := New(KitConfig{
		Storage: storage,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer kit.Close()

	if kit.Service == nil {
		t.Error("Expected non-nil Service")
	}
	if kit.Provider == nil {
		t.Error("Expected non-nil Provider")
	}
	if kit.LoadTool == nil {
		t.Error("Expected non-nil LoadTool")
	}
	if kit.PreloadTool == nil {
		t.Error("Expected non-nil PreloadTool")
	}
	if len(kit.Tools) != 2 {
		t.Errorf("Expected 2 tools, got %d", len(kit.Tools))
	}
	if kit.Plugin != nil {
		t.Error("Expected nil Plugin when compaction not configured")
	}
}

func TestNew_MissingStorage(t *testing.T) {
	_, err := New(KitConfig{})
	if err == nil {
		t.Fatal("New() expected error for missing storage, got nil")
	}
}

func TestNew_WithCompaction(t *testing.T) {
	storage := adapter.InMemory()
	defer storage.Close()

	strategy := &compaction.TruncationStrategy{RetainCount: 10}
	kit, err := New(KitConfig{
		Storage: storage,
		Compaction: &compaction.Config{
			Strategy:   strategy,
			MaxEvents:  100,
			KeepRecent: 10,
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer kit.Close()

	if kit.Plugin == nil {
		t.Error("Expected non-nil Plugin when compaction is configured")
	}
}

func TestNew_WithInvalidCompaction(t *testing.T) {
	storage := adapter.InMemory()
	defer storage.Close()

	_, err := New(KitConfig{
		Storage:    storage,
		Compaction: &compaction.Config{
			// Invalid: no strategy and no triggers
		},
	})
	if err == nil {
		t.Fatal("New() expected error for invalid compaction config, got nil")
	}
}

func TestNew_WithDeltaMode(t *testing.T) {
	storage := adapter.InMemory()
	defer storage.Close()

	kit, err := New(KitConfig{
		Storage:   storage,
		DeltaMode: true,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	defer kit.Close()

	// Verify service has delta mode enabled
	svc, ok := kit.Service.(*Service)
	if !ok {
		t.Fatal("Expected Service to be *Service")
	}
	if !svc.config.EnableDeltaMode {
		t.Error("Expected EnableDeltaMode to be true")
	}
}

func TestMemoryKit_Close(t *testing.T) {
	storage := adapter.InMemory()

	kit, err := New(KitConfig{
		Storage: storage,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	if err := kit.Close(); err != nil {
		t.Errorf("Close() error = %v", err)
	}
}
