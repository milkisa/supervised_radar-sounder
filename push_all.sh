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

# Detect remotes and set identity accordingly
if git remote -v | grep -q "github.com"; then
  echo "🔹 GitHub remote detected. Using personal identity..."
  git config user.name "Milkisa T. Yebasse"
  git config user.email "milkisatesfaye@gmail.com"
fi

if git remote -v | grep -q "bitbucket.org"; then
  echo "🔹 Bitbucket remote detected. Using university identity..."
  git config user.name "Milkisa T. Yebasse"
  git config user.email "milkisa.yebasse@unitn.it"
fi

# Add all changes
git add .

# Commit with your provided message
git commit -m "$1"

# Push to GitHub if remote exists
if git remote -v | grep -q "github"; then
  echo "🚀 Pushing to GitHub..."
  git push github main || echo "⚠️ GitHub push failed."
fi

# Push to Bitbucket if remote exists
if git remote -v | grep -q "bitbucket"; then
  echo "🚀 Pushing to Bitbucket..."
  git push bitbucket main || echo "⚠️ Bitbucket push failed."
fi

echo "✅ Done! Both remotes are up to date."
