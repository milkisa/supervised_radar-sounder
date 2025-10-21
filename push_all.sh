#!/bin/bash
# -----------------------------------------
# Auto add, commit, and push to GitHub + Bitbucket
# Usage: ./push_all.sh "commit message"
# -----------------------------------------

# Check if a commit message was provided
if [ -z "$1" ]; then
  echo "❗ Please provide a commit message."
  echo "Usage: ./push_all.sh \"your commit message\""
  exit 1
fi

# Add all changes
git add .

# Commit with your provided message
git commit -m "$1"

# Push to both remotes
echo "🚀 Pushing to GitHub..."
git push github main

echo "🚀 Pushing to Bitbucket..."
git push bitbucket main

echo "✅ Done! Both remotes are up to date."
