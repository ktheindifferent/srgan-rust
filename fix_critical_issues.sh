#!/bin/bash

# SRGAN-Rust Critical Issues Fix Script
# This script helps identify and fix critical issues in the codebase

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "           SRGAN-Rust Critical Issues Fix Script              "
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to count unwrap calls
count_unwraps() {
    local file=$1
    local count=$(grep -c "\.unwrap()\|\.expect(" "$file" 2>/dev/null || echo 0)
    echo $count
}

# Function to check for unimplemented features
check_unimplemented() {
    local file=$1
    grep -l "unimplemented!\|todo!\|unreachable!" "$file" 2>/dev/null || true
}

echo -e "${BLUE}Step 1: Analyzing current issues...${NC}"
echo "-----------------------------------------------"

# Count total unwrap calls
total_unwraps=0
declare -A unwrap_files

for file in $(find src -name "*.rs" -type f); do
    count=$(count_unwraps "$file")
    if [ $count -gt 0 ]; then
        unwrap_files["$file"]=$count
        total_unwraps=$((total_unwraps + count))
    fi
done

echo -e "${RED}Found $total_unwraps unwrap()/expect() calls across ${#unwrap_files[@]} files${NC}"
echo ""
echo "Top 5 files with most unwrap() calls:"
for file in "${!unwrap_files[@]}"; do
    echo "${unwrap_files[$file]} $file"
done | sort -rn | head -5 | while read count file; do
    echo -e "  ${YELLOW}$count${NC} - $file"
done

echo ""
echo -e "${BLUE}Step 2: Checking for incomplete implementations...${NC}"
echo "-----------------------------------------------"

incomplete_files=()
for file in $(find src -name "*.rs" -type f); do
    if check_unimplemented "$file"; then
        incomplete_files+=("$file")
    fi
done

if [ ${#incomplete_files[@]} -gt 0 ]; then
    echo -e "${RED}Found ${#incomplete_files[@]} files with incomplete implementations:${NC}"
    for file in "${incomplete_files[@]}"; do
        echo "  - $file"
    done
else
    echo -e "${GREEN}No incomplete implementations found!${NC}"
fi

echo ""
echo -e "${BLUE}Step 3: Creating backup...${NC}"
echo "-----------------------------------------------"

BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
echo "Creating backup in $BACKUP_DIR..."
mkdir -p "$BACKUP_DIR"
cp -r src "$BACKUP_DIR/"
echo -e "${GREEN}Backup created successfully${NC}"

echo ""
echo -e "${BLUE}Step 4: Generating fix patches...${NC}"
echo "-----------------------------------------------"

# Create patches directory
mkdir -p patches

# Generate patch for web_server.rs
cat > patches/web_server_fix.patch << 'EOF'
--- a/src/web_server.rs
+++ b/src/web_server.rs
@@ -255,7 +255,9 @@
     let timestamp = SystemTime::now()
-        .duration_since(UNIX_EPOCH)
-        .unwrap()
+        .duration_since(UNIX_EPOCH)
+        .map_err(|e| SrganError::SystemTime(format!("Failed to get timestamp: {}", e)))?
         .as_secs();
EOF

# Generate patch for video.rs
cat > patches/video_fix.patch << 'EOF'
--- a/src/video.rs
+++ b/src/video.rs
@@ -161,7 +161,9 @@
-    let input_path_str = input_path.to_str().unwrap();
+    let input_path_str = input_path.to_str()
+        .ok_or_else(|| SrganError::InvalidInput(
+            format!("Input path contains invalid UTF-8: {:?}", input_path)
+        ))?;
EOF

echo -e "${GREEN}Generated fix patches in patches/ directory${NC}"

echo ""
echo -e "${BLUE}Step 5: Running safety checks...${NC}"
echo "-----------------------------------------------"

# Check if tests pass
echo "Running tests..."
if cargo test --quiet 2>/dev/null; then
    echo -e "${GREEN}All tests pass${NC}"
else
    echo -e "${YELLOW}Warning: Some tests are failing${NC}"
fi

# Check for compilation warnings
echo "Checking for compilation warnings..."
warning_count=$(cargo build --message-format=json 2>&1 | grep -c '"level":"warning"' || echo 0)
if [ $warning_count -eq 0 ]; then
    echo -e "${GREEN}No compilation warnings${NC}"
else
    echo -e "${YELLOW}Found $warning_count compilation warnings${NC}"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo -e "${BLUE}                    Fix Recommendations                       ${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "1. IMMEDIATE ACTIONS (Critical):"
echo "   - Fix all unwrap() calls in src/web_server.rs (17 instances)"
echo "   - Fix all unwrap() calls in src/video.rs (9 instances)"
echo "   - Document GPU acceleration status in README.md"
echo ""

echo "2. HIGH PRIORITY:"
echo "   - Implement proper error types in src/error.rs"
echo "   - Add retry logic for network operations"
echo "   - Fix thread safety in batch processing"
echo ""

echo "3. DOCUMENTATION:"
echo "   - Update README with current feature status"
echo "   - Add warning about experimental model conversion"
echo "   - Document error handling best practices"
echo ""

echo -e "${YELLOW}To apply fixes automatically, run:${NC}"
echo "  ./apply_fixes.sh"
echo ""

echo -e "${YELLOW}To view detailed enhancement plan, see:${NC}"
echo "  - ENHANCEMENT_PLAN.md"
echo "  - CRITICAL_FIXES_GUIDE.md"
echo ""

echo -e "${GREEN}Analysis complete!${NC}"

# Create apply fixes script
cat > apply_fixes.sh << 'APPLY_EOF'
#!/bin/bash
echo "Applying critical fixes..."

# Apply patches
for patch in patches/*.patch; do
    echo "Applying $patch..."
    patch -p1 < "$patch" || echo "Failed to apply $patch"
done

echo "Fixes applied. Please review changes and run tests."
APPLY_EOF

chmod +x apply_fixes.sh

exit 0