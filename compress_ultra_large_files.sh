#!/bin/bash

# Script to upload project files while removing large .pt files (>50MB)
# This avoids GitHub's 100MB file size limit

set -e  # Exit on any error

echo "🚀 Starting ultra large files cleanup and upload process..."

# Configuration
PROJECT_DIR="/Users/tianrunliao/Desktop/pj2 submission"
REPO_URL="git@github.com:tianrunliao/Deep-Learning-Project2.git"
TEMP_DIR="/tmp/pj2_upload_$(date +%s)"

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ Error: Project directory not found: $PROJECT_DIR"
    exit 1
fi

echo "📁 Project directory: $PROJECT_DIR"
echo "🔗 Repository URL: $REPO_URL"
echo "📂 Temporary directory: $TEMP_DIR"

# Find and list large .pt files that will be removed
echo "\n🔍 Scanning for large .pt files (>50MB) to remove..."
find "$PROJECT_DIR" -name "*.pt" -size +50M -exec ls -lh {} \; | while read line; do
    echo "  📦 Will remove: $(echo $line | awk '{print $9}') ($(echo $line | awk '{print $5}'))"
done

# Count files to be removed
LARGE_PT_COUNT=$(find "$PROJECT_DIR" -name "*.pt" -size +50M | wc -l | tr -d ' ')
echo "\n📊 Found $LARGE_PT_COUNT large .pt files to remove"

if [ "$LARGE_PT_COUNT" -gt 0 ]; then
    echo "\n⚠️  These files will be permanently deleted from the upload (local files remain unchanged)"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Upload cancelled by user"
        exit 1
    fi
fi

# Create temporary directory
echo "\n📂 Creating temporary directory..."
mkdir -p "$TEMP_DIR"

# Copy project files to temporary directory
echo "📋 Copying project files..."
cp -r "$PROJECT_DIR"/* "$TEMP_DIR/"

# Remove large .pt files from temporary directory
if [ "$LARGE_PT_COUNT" -gt 0 ]; then
    echo "\n🗑️  Removing large .pt files from upload..."
    find "$TEMP_DIR" -name "*.pt" -size +50M -delete
    echo "✅ Removed $LARGE_PT_COUNT large .pt files"
fi

# Create documentation about removed files
echo "\n📝 Creating documentation about removed files..."
cat > "$TEMP_DIR/LARGE_FILES_REMOVED.md" << EOF
# Large Files Removed

This project originally contained large PyTorch checkpoint files (.pt) that exceeded GitHub's 100MB file size limit.

## Removed Files
The following large .pt files (>50MB) were removed from this upload:

EOF

find "$PROJECT_DIR" -name "*.pt" -size +50M -exec ls -lh {} \; | while read line; do
    filename=$(echo $line | awk '{print $9}' | sed "s|$PROJECT_DIR/||")
    size=$(echo $line | awk '{print $5}')
    echo "- \`$filename\` ($size)" >> "$TEMP_DIR/LARGE_FILES_REMOVED.md"
done

cat >> "$TEMP_DIR/LARGE_FILES_REMOVED.md" << EOF

## Note
- These files contain trained model checkpoints
- They are not required for code evaluation
- All source code and configuration files are included
- Results and analysis files are preserved

## Reproduction
To reproduce the results, run the training scripts provided in the code directory.
EOF

# Change to temporary directory
cd "$TEMP_DIR"

# Initialize git repository
echo "\n🔧 Initializing Git repository..."
git init
git remote add origin "$REPO_URL"

# Create .gitignore for any remaining large files
echo "\n📋 Creating .gitignore..."
cat > .gitignore << EOF
# Large files
*.pt
*.pth
*.ckpt
*.model

# Temporary files
*.tmp
*.temp
__pycache__/
*.pyc
*.pyo

# System files
.DS_Store
Thumbs.db
EOF

# Add all files
echo "\n➕ Adding files to Git..."
git add .

# Commit
echo "\n💾 Committing files..."
git commit -m "Upload Deep Learning Project 2 - Large checkpoint files removed

- Removed .pt files larger than 50MB to comply with GitHub limits
- All source code, configurations, and results preserved
- See LARGE_FILES_REMOVED.md for details"

# Push to GitHub
echo "\n🚀 Pushing to GitHub..."
git push -u origin main

echo "\n✅ Upload completed successfully!"
echo "\n📊 Summary:"
echo "  - Repository: https://github.com/tianrunliao/Deep-Learning-Project2"
echo "  - Large .pt files removed: $LARGE_PT_COUNT"
echo "  - Documentation: LARGE_FILES_REMOVED.md"
echo "  - Local files: Unchanged"

# Cleanup
echo "\n🧹 Cleaning up temporary directory..."
rm -rf "$TEMP_DIR"

echo "\n🎉 All done! Your project is now available on GitHub."