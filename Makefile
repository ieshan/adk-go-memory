.PHONY: test test-race check build vet clean test-sqlite setup

# Setup development workspace
setup:
	@echo "Initializing project..."
	go work init . ./adapter/sqlite ./examples
	@echo "Project initialized."

# Default test run (uses map-based MemoryStorage, no CGO required)
test:
	go test -v ./...

# Run SQLite adapter tests (requires CGO)
test-sqlite:
	cd adapter/sqlite && CGO_ENABLED=1 go test -v -tags=sqlite_fts5 ./...

# Run with race detection
test-race:
	go test -v -race ./...
	cd adapter/sqlite && CGO_ENABLED=1 go test -v -race -tags=sqlite_fts5 ./...

# Build all packages
build:
	go build ./...
	cd adapter/sqlite && CGO_ENABLED=1 go build -tags=sqlite_fts5 ./...

# Run go vet
vet:
	go vet ./...
	cd adapter/sqlite && CGO_ENABLED=1 go vet -tags=sqlite_fts5 ./...

# Run go fmt
fmt:
	go fmt ./...
	cd adapter/sqlite && go fmt ./...

# Check for common issues
check: vet test test-sqlite
	@echo "All checks passed"

# Run integration tests (placeholder for future integration tests)
integration-test:
	CGO_ENABLED=1 go test -v -tags=\"sqlite_fts5 integration\" ./...

# Coverage report (root + sqlite adapter)
coverage:
	go test -coverprofile=coverage.out ./...
	go tool cover -func=coverage.out
	cd adapter/sqlite && CGO_ENABLED=1 go test -coverprofile=coverage.out -tags=sqlite_fts5 ./...
	cd adapter/sqlite && go tool cover -func=coverage.out

# Clean build artifacts
clean:
	rm -f coverage.out
	rm -f adapter/sqlite/coverage.out
