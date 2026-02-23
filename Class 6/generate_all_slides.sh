#!/bin/bash
# Generate all Class 6 (Week 6) slides with playful academic cartoon style
# Uses DALL-E 3 model

echo "=========================================="
echo "CLASS 6 - WEEK 6 SLIDE GENERATOR"
echo "Coding with AI"
echo "=========================================="
echo "This will generate images for all 27 slides"
echo "Estimated time: ~15 minutes"
echo "Estimated cost: ~\$2.16 (27 slides × \$0.08)"
echo "=========================================="
echo ""

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    echo "Please run: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Confirm before proceeding
read -p "Continue with batch generation? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Starting batch generation..."
echo "=========================================="

# Counter for progress
total=27
completed=0
failed=0

# Generate all 27 slides
for i in {1..27}; do
    echo ""
    echo "[$((completed+failed+1))/$total] Generating slide $i..."
    python3 generate_any_slide.py $i
    
    if [ $? -eq 0 ]; then
        ((completed++))
        echo "✅ Slide $i complete ($completed/$total successful)"
    else
        ((failed++))
        echo "❌ Failed to generate slide $i ($failed failures)"
    fi
    
    # Wait between calls to avoid rate limiting (3 seconds)
    if [ $i -lt 27 ]; then
        echo "⏳ Waiting 3 seconds before next request..."
        sleep 3
    fi
    
    # Commit every 5 slides to save progress
    if [ $((i % 5)) -eq 0 ]; then
        echo ""
        echo "📦 Committing slides batch (up to slide $i)..."
        git add slide_images/*.png slide_*_image_url.txt 2>/dev/null
        if git diff --staged --quiet; then
            echo "ℹ️  No new files to commit"
        else
            git commit -m "Generate Week 6 slides batch (slides 1-$i)"
            echo "✅ Committed"
        fi
    fi
done

# Final commit
echo ""
echo "📦 Final commit..."
git add slide_images/*.png slide_*_image_url.txt 2>/dev/null
if git diff --staged --quiet; then
    echo "ℹ️  No new files to commit"
else
    git commit -m "Complete all Week 6 slide images - playful academic cartoon style"
    echo "✅ Final commit complete"
fi

echo ""
echo "=========================================="
echo "✨ BATCH GENERATION COMPLETE!"
echo "=========================================="
echo "Successfully generated: $completed/$total slides"
echo "Failed: $failed/$total slides"
echo "Location: $SCRIPT_DIR/slide_images/"
echo ""

if [ $failed -gt 0 ]; then
    echo "⚠️  Some slides failed to generate."
    echo "You can retry individual slides with:"
    echo "  python3 generate_any_slide.py <slide_number>"
    echo ""
fi

if [ $completed -eq $total ]; then
    echo "🎉 All slides generated successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Review images in slide_images/ folder"
    echo "2. Push to GitHub: git push"
    echo "3. Use images in your slide presentations"
    echo ""
fi

exit 0
