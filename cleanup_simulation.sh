#!/bin/bash

# Tidal Protocol Simulation Cleanup Script
# This script helps clean up large JSON result files and Python cache

echo "🧹 Tidal Protocol Simulation Cleanup"
echo "===================================="

# Function to show file sizes
show_current_size() {
    echo "📊 Current directory size:"
    du -sh .
    echo ""
    echo "📁 Results directory breakdown:"
    du -sh tidal_protocol_sim/results/* 2>/dev/null || echo "No results directories found"
    echo ""
}

# Function to clean JSON results
clean_json_results() {
    echo "🗂️  Large JSON files found:"
    find . -name "*.json" -size +10M -exec ls -lah {} \; 2>/dev/null
    echo ""
    
    read -p "❓ Do you want to remove large JSON result files (>10MB)? [y/N]: " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing large JSON files..."
        find . -name "*.json" -size +10M -delete
        echo "✅ Large JSON files removed!"
    else
        echo "⏭️  Skipping JSON cleanup"
    fi
    echo ""
}

# Function to clean Python cache
clean_python_cache() {
    echo "🐍 Python cache cleanup:"
    
    # Find and show __pycache__ directories
    pycache_dirs=$(find . -type d -name "__pycache__" 2>/dev/null)
    if [ -n "$pycache_dirs" ]; then
        echo "📁 Found __pycache__ directories:"
        echo "$pycache_dirs"
        echo ""
        
        read -p "❓ Remove all __pycache__ directories? [y/N]: " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
            echo "✅ Python cache directories removed!"
        else
            echo "⏭️  Skipping Python cache cleanup"
        fi
    else
        echo "✅ No __pycache__ directories found"
    fi
    
    # Find and show .pyc files
    pyc_files=$(find . -name "*.pyc" 2>/dev/null)
    if [ -n "$pyc_files" ]; then
        echo ""
        echo "📄 Found .pyc files:"
        echo "$pyc_files" | head -10
        [ $(echo "$pyc_files" | wc -l) -gt 10 ] && echo "... and more"
        echo ""
        
        read -p "❓ Remove all .pyc files? [y/N]: " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            find . -name "*.pyc" -delete
            echo "✅ .pyc files removed!"
        else
            echo "⏭️  Skipping .pyc cleanup"
        fi
    else
        echo "✅ No .pyc files found"
    fi
    echo ""
}

# Function to clean specific result directories
clean_result_directories() {
    echo "📂 Simulation results cleanup:"
    
    if [ -d "tidal_protocol_sim/results" ]; then
        echo "📁 Results directories:"
        ls -lah tidal_protocol_sim/results/
        echo ""
        
        read -p "❓ Do you want to remove all simulation results? [y/N]: " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf tidal_protocol_sim/results/*
            echo "✅ All simulation results removed!"
        else
            echo "⏭️  Keeping simulation results"
            
            # Offer to clean just the large comprehensive analysis
            if [ -d "tidal_protocol_sim/results/Comprehensive_HT_vs_Aave_Analysis" ]; then
                echo ""
                read -p "❓ Remove just the large Comprehensive_HT_vs_Aave_Analysis directory (56MB)? [y/N]: " -n 1 -r
                echo ""
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    rm -rf tidal_protocol_sim/results/Comprehensive_HT_vs_Aave_Analysis
                    echo "✅ Comprehensive analysis results removed!"
                fi
            fi
        fi
    else
        echo "✅ No results directory found"
    fi
    echo ""
}

# Main execution
echo "📏 Initial size assessment:"
show_current_size

# Menu-driven cleanup
echo "🛠️  Cleanup Options:"
echo "1. Clean large JSON files (>10MB)"
echo "2. Clean Python cache (__pycache__, .pyc)"
echo "3. Clean simulation results directories"
echo "4. Clean everything"
echo "5. Show current size only"
echo ""

read -p "❓ Choose an option (1-5): " -n 1 -r
echo ""
echo ""

case $REPLY in
    1)
        clean_json_results
        ;;
    2)
        clean_python_cache
        ;;
    3)
        clean_result_directories
        ;;
    4)
        clean_json_results
        clean_python_cache
        clean_result_directories
        ;;
    5)
        echo "📊 Current size only - no cleanup performed"
        ;;
    *)
        echo "❌ Invalid option. Exiting."
        exit 1
        ;;
esac

echo "📏 Final size assessment:"
show_current_size

echo "✨ Cleanup complete!"
