#!/bin/bash
set -e

REPO_PATH="${1:-/repo}"
GIT_REPO_URL="${GIT_REPO_URL}"
GIT_BRANCH="${GIT_BRANCH:-main}"
GIT_TOKEN="${GIT_TOKEN:-}"

if [ -z "$GIT_REPO_URL" ]; then
    echo "ERROR: GIT_REPO_URL is not set"
    exit 1
fi

echo "Repository sync script"
echo "REPO_PATH: $REPO_PATH"
echo "GIT_REPO_URL: $GIT_REPO_URL"
echo "GIT_BRANCH: $GIT_BRANCH"

# Prepare URL with token if provided
GIT_URL="$GIT_REPO_URL"
if [ -n "$GIT_TOKEN" ]; then
    # Replace https:// with https://${GIT_TOKEN}@
    GIT_URL=$(echo "$GIT_REPO_URL" | sed "s|https://|https://${GIT_TOKEN}@|")
    echo "Using token authentication for private repository"
fi

# Check if repository already exists
if [ -d "$REPO_PATH/.git" ]; then
    echo "Repository exists, updating..."
    cd "$REPO_PATH"
    
    # Fetch all branches
    git fetch --all
    
    # Reset to remote branch
    git reset --hard "origin/${GIT_BRANCH}"
    
    # Clean untracked files
    git clean -fd
    
    echo "Repository updated successfully"
else
    echo "Repository does not exist, cloning..."
    
    # Clone repository
    git clone --branch "$GIT_BRANCH" --single-branch "$GIT_URL" "$REPO_PATH"
    
    echo "Repository cloned successfully"
fi

# Show current commit
cd "$REPO_PATH"
echo "Current commit: $(git rev-parse HEAD)"
echo "Current branch: $(git rev-parse --abbrev-ref HEAD)"
