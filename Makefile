.PHONY: test test-race check build vet clean

# Default test run (requires CGO for sqlite-vec)
test:
	CGO_ENABLED=1 go test -v -tags=sqlite_fts5 ./...

# Run with race detection
test-race:
	CGO_ENABLED=1 go test -v -race -tags=sqlite_fts5 ./...

# Build all packages
build:
	CGO_ENABLED=1 go build -tags=sqlite_fts5 ./...

# Run go vet
vet:
	CGO_ENABLED=1 go vet -tags=sqlite_fts5 ./...

# Check for common issues
check: vet test
	@echo "All checks passed"

# Run integration tests (placeholder for future integration tests)
integration-test:
	CGO_ENABLED=1 go test -v -tags=\"sqlite_fts5 integration\" ./...

# Coverage report
coverage:
	CGO_ENABLED=1 go test -tags=sqlite_fts5 -coverprofile=coverage.out ./...
	go tool cover -func=coverage.out

# Clean build artifacts
clean:
	rm -f coverage.out
