#!/bin/bash

# Setup Neo4j for CoTKG-IDS
echo "Setting up Neo4j for CoTKG-IDS..."

# Check if Neo4j is installed
if ! command -v neo4j &> /dev/null; then
    echo "Neo4j not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install neo4j
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
        echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
        sudo apt-get update
        sudo apt-get install neo4j
    else
        echo "Unsupported OS. Please install Neo4j manually."
        exit 1
    fi
fi

# Start Neo4j service
echo "Starting Neo4j service..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    brew services start neo4j
else
    sudo service neo4j start
fi

# Wait for Neo4j to start
echo "Waiting for Neo4j to start..."
sleep 5

# Set password
neo4j-admin set-initial-password neo4jneo4j

echo "Neo4j setup complete!"
