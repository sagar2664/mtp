#!/bin/bash
# Convert dthesis.md → dthesis.docx with embedded figures and equations
# Usage: bash convert_thesis.sh

PANDOC="/tmp/pandoc-3.6.4-arm64/bin/pandoc"

if [ ! -f "$PANDOC" ]; then
    echo "Pandoc not found. Downloading ARM64 binary..."
    curl -L "https://github.com/jgm/pandoc/releases/download/3.6.4/pandoc-3.6.4-arm64-macOS.zip" -o /tmp/pandoc-arm.zip
    cd /tmp && unzip -o pandoc-arm.zip && cd -
fi

echo "Converting dthesis.md → dthesis.docx ..."
$PANDOC dthesis.md \
    -o dthesis.docx \
    --toc \
    --number-sections \
    --standalone \
    --resource-path=.

if [ $? -eq 0 ]; then
    echo "✅ Success! Output: dthesis.docx ($(ls -lh dthesis.docx | awk '{print $5}'))"
    echo "   open dthesis.docx"
else
    echo "❌ Conversion failed."
fi
