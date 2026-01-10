#!/bin/bash
# Build and run the green agent in Docker
# Replace 'your-registry/your-repo:tag' with your actual Docker image
docker build -t your-registry/agentbeats-green:v1 .
docker run --rm -p 8080:8080 your-registry/agentbeats-green:v1 --host 0.0.0.0 --port 8080 --card-url http://example.com/card