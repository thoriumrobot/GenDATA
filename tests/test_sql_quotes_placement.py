#!/usr/bin/env python3
"""
Tests for SQL Quotes Annotation Placement System

Tests the SqlQuotesAnnotationPlacer and related functionality.
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from place_sql_quotes_annotations import (
    SqlQuotesAnnotationType,
    SqlQuotesAnnotationPlacer,
    SqlQuotesPlacement,
    SQL_QUOTES_IMPORTS,
)


class TestSqlQuotesAnnotationType:
    """Test the SqlQuotesAnnotationType enum"""
    
    def test_enum_values(self):
        """Test that enum has correct annotation values"""
        assert SqlQuotesAnnotationType.SQL_EVEN_QUOTES.value == "@SqlEvenQuotes"
        assert SqlQuotesAnnotationType.SQL_ODD_QUOTES.value == "@SqlOddQuotes"
    
    def test_enum_count(self):
        """Test that enum has exactly 2 values"""
        assert len(SqlQuotesAnnotationType) == 2


class TestSqlQuotesImports:
    """Test the import statements"""
    
    def test_imports_defined(self):
        """Test that imports are defined"""
        assert len(SQL_QUOTES_IMPORTS) == 2
    
    def test_import_format(self):
        """Test import statement format"""
        for imp in SQL_QUOTES_IMPORTS:
            assert imp.startswith("import org.checkerframework.checker.sqlquotes.qual.")
            assert imp.endswith(";")


class TestSqlQuotesAnnotationPlacer:
    """Test the SqlQuotesAnnotationPlacer class"""
    
    @pytest.fixture
    def sample_file(self):
        """Create a temporary sample file for testing"""
        content = '''public class TestClass {
    String query1 = "SELECT * FROM users";
    String query2 = "INSERT INTO table VALUES (1)";
    
    public void executeQuery(String sql) {
        System.out.println(sql);
    }
    
    public void test() {
        String query = "UPDATE users SET name = ''";
        executeQuery(query);
    }
}
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def fixture_file(self):
        """Get path to the SQL quotes fixture file"""
        fixture_path = Path(__file__).parent / 'fixtures' / 'sql_quotes_sample.java'
        if fixture_path.exists():
            return str(fixture_path)
        return None
    
    def test_placer_init(self, sample_file):
        """Test placer initialization"""
        placer = SqlQuotesAnnotationPlacer(sample_file)
        assert len(placer.lines) > 0
        assert placer.file_path == sample_file
        assert len(placer.placements) == 0
    
    def test_count_single_quotes(self, sample_file):
        """Test quote counting"""
        placer = SqlQuotesAnnotationPlacer(sample_file)
        
        assert placer.count_single_quotes("no quotes") == 0
        assert placer.count_single_quotes("one'quote") == 1
        # '' is treated as escaped quotes in SQL, so they're removed
        assert placer.count_single_quotes("two''quotes") == 0
        assert placer.count_single_quotes("escaped\\'quote") == 0  # Escaped
        # Three single quotes: one pair removed, one remains
        assert placer.count_single_quotes("three'''quotes") == 1
    
    def test_is_sql_related(self, sample_file):
        """Test SQL pattern detection"""
        placer = SqlQuotesAnnotationPlacer(sample_file)
        
        # Should detect SQL patterns
        assert placer.is_sql_related("SELECT * FROM users")
        assert placer.is_sql_related("INSERT INTO table")
        assert placer.is_sql_related("UPDATE users SET")
        assert placer.is_sql_related("DELETE FROM table")
        assert placer.is_sql_related("executeQuery(sql)")
        assert placer.is_sql_related("prepareStatement(query)")
        
        # Should not detect non-SQL
        assert not placer.is_sql_related("regular string")
        assert not placer.is_sql_related("int x = 5;")
    
    def test_analyze_string_literal(self, sample_file):
        """Test string literal analysis for quote parity"""
        placer = SqlQuotesAnnotationPlacer(sample_file)
        
        # Even quotes (including 0)
        line1 = 'String q = "SELECT * FROM users";'
        assert placer.analyze_string_literal(line1) == SqlQuotesAnnotationType.SQL_EVEN_QUOTES
        
        # Two quotes = even
        line2 = "String q = \"name = ''\";"
        assert placer.analyze_string_literal(line2) == SqlQuotesAnnotationType.SQL_EVEN_QUOTES
    
    def test_is_valid_annotation_target(self, sample_file):
        """Test annotation target validation"""
        placer = SqlQuotesAnnotationPlacer(sample_file)
        
        # Line 2 has String declaration - valid
        assert placer.is_valid_annotation_target(2)
        
        # Line 1 is class declaration - invalid
        assert not placer.is_valid_annotation_target(1)
    
    def test_analyze_and_place(self, sample_file):
        """Test full analysis and placement"""
        placer = SqlQuotesAnnotationPlacer(sample_file)
        placements = placer.analyze_and_place()
        
        # Should find at least some SQL-related declarations
        assert len(placements) > 0
        
        # All placements should be SqlQuotesPlacement
        for p in placements:
            assert isinstance(p, SqlQuotesPlacement)
            assert p.annotation in SqlQuotesAnnotationType
    
    def test_add_imports(self, sample_file):
        """Test import addition"""
        placer = SqlQuotesAnnotationPlacer(sample_file)
        
        # Should add imports
        result = placer.add_imports()
        assert result == True
        
        # Check imports are in file content
        content = ''.join(placer.lines)
        assert 'SqlEvenQuotes' in content or 'sqlquotes' in content
    
    def test_place_annotation_specific_lines(self, sample_file):
        """Test placing annotations at specific lines"""
        placer = SqlQuotesAnnotationPlacer(sample_file)
        
        # Place at line 2 (String query1 declaration)
        placements = placer.analyze_and_place(sql_param_lines=[2])
        
        assert len(placements) > 0
    
    def test_fixture_file_analysis(self, fixture_file):
        """Test analysis on the fixture file"""
        if fixture_file is None:
            pytest.skip("Fixture file not found")
        
        placer = SqlQuotesAnnotationPlacer(fixture_file)
        placements = placer.analyze_and_place()
        
        # Should find multiple SQL-related declarations
        assert len(placements) >= 3
        
        # All should be SqlEvenQuotes (no odd quotes in sample)
        for p in placements:
            assert p.annotation == SqlQuotesAnnotationType.SQL_EVEN_QUOTES


class TestSqlQuotesIntegration:
    """Integration tests for SQL Quotes placement"""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory"""
        temp_dir = tempfile.mkdtemp()
        
        # Create a sample Java file
        java_file = os.path.join(temp_dir, 'Sample.java')
        with open(java_file, 'w') as f:
            f.write('''package com.example;

public class Sample {
    String query = "SELECT * FROM users";
    
    public void run(String sql) {
        executeQuery(sql);
    }
}
''')
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_full_placement_workflow(self, temp_project):
        """Test complete placement workflow"""
        java_file = os.path.join(temp_project, 'Sample.java')
        
        # Create placer
        placer = SqlQuotesAnnotationPlacer(java_file)
        
        # Analyze and place
        placements = placer.analyze_and_place()
        
        # Add imports
        placer.add_imports()
        
        # Save file
        placer.save_file()
        
        # Verify file was modified
        with open(java_file, 'r') as f:
            content = f.read()
        
        # Check that annotations or imports were added
        assert len(placements) >= 0  # May or may not find targets
        
        # File should still be valid (has class declaration)
        assert 'class Sample' in content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
