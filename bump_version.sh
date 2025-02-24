#!/bin/bash

# Script to bump version numbers

# Get current version from setup.py
current_version=$(grep "version=" setup.py | cut -d'"' -f2)

# Parse version components
IFS='.' read -r -a version_parts <<< "$current_version"
major="${version_parts[0]}"
minor="${version_parts[1]}"
patch="${version_parts[2]}"

case "$1" in
    "major")
        major=$((major + 1))
        minor=0
        patch=0
        ;;
    "minor")
        minor=$((minor + 1))
        patch=0
        ;;
    "patch")
        patch=$((patch + 1))
        ;;
    *)
        echo "Usage: $0 {major|minor|patch}"
        exit 1
        ;;
esac

new_version="$major.$minor.$patch"

# Update version in setup.py
sed -i.bak "s/version=\".*\"/version=\"$new_version\"/" setup.py
rm setup.py.bak

echo "Version bumped from $current_version to $new_version"

# Create git tag
git add setup.py
git commit -m "Bump version to $new_version"
git tag -a "v$new_version" -m "Version $new_version"
