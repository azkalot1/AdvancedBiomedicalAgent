#!/bin/bash
# Download BindingDB TSV file and prepare for ingestion

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data/bindingdb"

echo "==============================================="
echo "BindingDB TSV Download"
echo "==============================================="
echo ""

# Create data directory
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Download TSV file (~500 MB zipped, ~3 GB unzipped)
echo "📥 Downloading BindingDB TSV file..."
echo "Source: https://www.bindingdb.org"
echo ""

# Direct download (use /rwd/bind/downloads/ path)
echo "Attempting direct download..."
wget --progress=bar:force:noscroll \
     --user-agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36" \
     "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202510_tsv.zip" \
     -O BindingDB_All_202510_tsv.zip

# Check if download was successful (should be > 100 MB)
FILE_SIZE=$(stat -f%z BindingDB_All_202510_tsv.zip 2>/dev/null || stat -c%s BindingDB_All_202510_tsv.zip 2>/dev/null)
if [ "$FILE_SIZE" -lt 100000000 ]; then
    echo ""
    echo "⚠️  Downloaded file is too small ($FILE_SIZE bytes)"
    echo "This usually means the direct download failed."
    echo ""
    echo "Please download manually:"
    echo "  1. Go to: https://www.bindingdb.org/bind/downloads/"
    echo "  2. Click on: BindingDB_All_202510_tsv.zip"
    echo "  3. Save to: $DATA_DIR/BindingDB_All_202510_tsv.zip"
    echo ""
    exit 1
fi

echo ""
echo "✓ Download complete!"
echo ""

# Unzip
echo "📦 Extracting TSV file..."
unzip -o BindingDB_All_202510_tsv.zip

# Find the actual TSV file (name might vary)
TSV_FILE=$(find . -maxdepth 1 -name "*.tsv" | head -1)

if [ -z "$TSV_FILE" ]; then
    echo "❌ Error: No TSV file found after extraction"
    exit 1
fi

echo "✓ Extracted: $TSV_FILE"
echo ""

# Show file info
echo "📊 File Information:"
echo "-------------------"
ls -lh "$TSV_FILE"
echo ""

# Show first few lines (header)
echo "📋 TSV File Structure (first 3 lines):"
echo "----------------------------------------"
head -n 3 "$TSV_FILE" | cut -f1-10
echo "... (showing first 10 columns only)"
echo ""

# Count lines (approximate)
echo "📈 Counting records (this may take a minute)..."
LINE_COUNT=$(wc -l < "$TSV_FILE")
RECORD_COUNT=$((LINE_COUNT - 1))  # Subtract header
echo "Total records: ~$RECORD_COUNT"
echo ""

echo "==============================================="
echo "✓ BindingDB TSV ready for import!"
echo "==============================================="
echo ""
echo "Location: $DATA_DIR"
echo "File: $TSV_FILE"
echo ""
echo "Next steps:"
echo "  1. Inspect columns: head -n 1 $TSV_FILE | tr '\t' '\n' | nl"
echo "  2. Run import: python -m bioagent.data.ingest.import_bindingdb_tsv"
echo ""

