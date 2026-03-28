#!/bin/bash

echo "🔍 Detecting OS..."

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Linux detected"

    # Detect package manager
    if command -v pacman &> /dev/null; then
        echo "📦 Installing dependencies (Arch)..."
        sudo pacman -Sy --noconfirm freeglut glew mesa
    elif command -v apt &> /dev/null; then
        echo "📦 Installing dependencies (Ubuntu/Debian)..."
        sudo apt update
        sudo apt install -y freeglut3-dev libglew-dev mesa-utils
    else
        echo "Unsupported package manager"
        exit 1
    fi

    echo "Compiling..."
    gcc julia_viewer.c -lglut -lGLEW -lGL -lGLU -lm -o julia_viewer

    if [ $? -eq 0 ]; then
        echo "Build complete!"
        echo "Run with: ./julia_viewer"
    else
        echo "Compilation failed"
    fi

else
    echo "This script is for Linux. Use compile.bat on Windows."
fi