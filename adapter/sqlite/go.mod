module github.com/ieshan/adk-go-memory/adapter/sqlite

go 1.26

replace github.com/ieshan/adk-go-memory => ../../

require (
	github.com/asg017/sqlite-vec-go-bindings v0.1.6
	github.com/ieshan/adk-go-memory v0.0.0
	github.com/mattn/go-sqlite3 v1.14.42
)
