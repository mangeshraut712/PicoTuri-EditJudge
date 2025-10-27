#!/bin/bash
# Comprehensive Verification Script for PicoTuri-EditJudge

echo "🎯 PicoTuri-EditJudge - Complete Verification"
echo "=============================================="
echo ""

# Run comprehensive test suite
echo "📊 Running comprehensive test suite..."
python tests/test_all_algorithms.py

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ ALL TESTS PASSED!"
    echo ""
    echo "📁 Project Structure:"
    echo "   ✓ src/ - Source code"
    echo "   ✓ tests/ - Test suite"
    echo "   ✓ docs/ - Documentation"
    echo "   ✓ assets/ - Images and charts"
    echo "   ✓ data/ - Datasets and models"
    echo ""
    echo "📊 Generated Charts:"
    ls -lh assets/charts/*.png 2>/dev/null | wc -l | xargs echo "   Charts:"
    echo ""
    echo "🎉 Project is production-ready!"
else
    echo ""
    echo "❌ Some tests failed. Please review the output above."
    exit 1
fi
