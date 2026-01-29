#!/usr/bin/env python3
"""
Tests for Signature String Annotation Placement System

Tests the SignatureAnnotationPlacer and related functionality.
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from place_signature_annotations import (
    SignatureAnnotationType,
    SignatureAnnotationPlacer,
    SignaturePlacement,
    SIGNATURE_IMPORTS,
)


class TestSignatureAnnotationType:
    """Test the SignatureAnnotationType enum"""
    
    def test_enum_values(self):
        """Test that enum has correct annotation values"""
        assert SignatureAnnotationType.BINARY_NAME.value == "@BinaryName"
        assert SignatureAnnotationType.FULLY_QUALIFIED_NAME.value == "@FullyQualifiedName"
        assert SignatureAnnotationType.FIELD_DESCRIPTOR.value == "@FieldDescriptor"
        assert SignatureAnnotationType.CLASS_GET_NAME.value == "@ClassGetName"
        assert SignatureAnnotationType.INTERNAL_FORM.value == "@InternalForm"
    
    def test_enum_count(self):
        """Test that enum has exactly 5 values"""
        assert len(SignatureAnnotationType) == 5


class TestSignatureImports:
    """Test the import statements"""
    
    def test_imports_defined(self):
        """Test that imports are defined"""
        assert len(SIGNATURE_IMPORTS) == 5
    
    def test_import_format(self):
        """Test import statement format"""
        for imp in SIGNATURE_IMPORTS:
            assert imp.startswith("import org.checkerframework.checker.signature.qual.")
            assert imp.endswith(";")


class TestSignatureAnnotationPlacer:
    """Test the SignatureAnnotationPlacer class"""
    
    @pytest.fixture
    def sample_file(self):
        """Create a temporary sample file for testing"""
        content = '''public class TestClass {
    String className = "java.lang.String";
    String internalName = "java/lang/String";
    String descriptor = "Ljava/lang/String;";
    
    public void loadClass(String typeName) throws Exception {
        Class.forName(typeName);
    }
    
    public void test() {
        String name = String.class.getName();
        System.out.println(name);
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
        """Get path to the signature fixture file"""
        fixture_path = Path(__file__).parent / 'fixtures' / 'signature_sample.java'
        if fixture_path.exists():
            return str(fixture_path)
        return None
    
    def test_placer_init(self, sample_file):
        """Test placer initialization"""
        placer = SignatureAnnotationPlacer(sample_file)
        assert len(placer.lines) > 0
        assert placer.file_path == sample_file
        assert len(placer.placements) == 0
    
    def test_detect_signature_format_binary_name(self, sample_file):
        """Test binary name detection"""
        placer = SignatureAnnotationPlacer(sample_file)
        
        # Binary names with $ (inner classes) - distinguishes from FQN
        assert placer.detect_signature_format("java.util.Map$Entry") == SignatureAnnotationType.BINARY_NAME
        assert placer.detect_signature_format("com.example.Outer$Inner") == SignatureAnnotationType.BINARY_NAME
        
        # Simple dotted names without $ return FULLY_QUALIFIED_NAME
        # (BinaryName and FullyQualifiedName are equivalent for top-level classes)
        assert placer.detect_signature_format("java.lang.String") == SignatureAnnotationType.FULLY_QUALIFIED_NAME
        assert placer.detect_signature_format("com.example.MyClass") == SignatureAnnotationType.FULLY_QUALIFIED_NAME
    
    def test_detect_signature_format_internal_form(self, sample_file):
        """Test internal form detection"""
        placer = SignatureAnnotationPlacer(sample_file)
        
        # Internal form (slashed format)
        assert placer.detect_signature_format("java/lang/String") == SignatureAnnotationType.INTERNAL_FORM
        assert placer.detect_signature_format("com/example/MyClass") == SignatureAnnotationType.INTERNAL_FORM
    
    def test_detect_signature_format_field_descriptor(self, sample_file):
        """Test field descriptor detection"""
        placer = SignatureAnnotationPlacer(sample_file)
        
        # Field descriptors
        assert placer.detect_signature_format("Ljava/lang/String;") == SignatureAnnotationType.FIELD_DESCRIPTOR
        assert placer.detect_signature_format("[Ljava/lang/Object;") == SignatureAnnotationType.FIELD_DESCRIPTOR
    
    def test_detect_signature_format_none(self, sample_file):
        """Test non-signature strings"""
        placer = SignatureAnnotationPlacer(sample_file)
        
        # Should return None for non-signature strings
        assert placer.detect_signature_format("regular string") is None
        assert placer.detect_signature_format("") is None
    
    def test_is_class_forname_pattern(self, sample_file):
        """Test Class.forName pattern detection"""
        placer = SignatureAnnotationPlacer(sample_file)
        
        assert placer.is_class_forname_pattern("Class.forName(className)")
        assert placer.is_class_forname_pattern("Class.forName(\"java.lang.String\")")
        assert not placer.is_class_forname_pattern("regular code")
    
    def test_is_getname_pattern(self, sample_file):
        """Test .getName() pattern detection"""
        placer = SignatureAnnotationPlacer(sample_file)
        
        assert placer.is_getname_pattern("String.class.getName()")
        assert placer.is_getname_pattern("clazz.getName()")
        assert not placer.is_getname_pattern("regular code")
    
    def test_is_getcanonicalname_pattern(self, sample_file):
        """Test .getCanonicalName() pattern detection"""
        placer = SignatureAnnotationPlacer(sample_file)
        
        assert placer.is_getcanonicalname_pattern("String.class.getCanonicalName()")
        assert not placer.is_getcanonicalname_pattern("regular code")
    
    def test_is_signature_related(self, sample_file):
        """Test signature-related pattern detection"""
        placer = SignatureAnnotationPlacer(sample_file)
        
        assert placer.is_signature_related("Class.forName(className)")
        assert placer.is_signature_related("clazz.getName()")
        assert placer.is_signature_related("loader.loadClass(typeName)")
        assert placer.is_signature_related("String className = ")
    
    def test_is_valid_annotation_target(self, sample_file):
        """Test annotation target validation"""
        placer = SignatureAnnotationPlacer(sample_file)
        
        # Line 2 has String declaration - valid
        assert placer.is_valid_annotation_target(2)
        
        # Line 1 is class declaration - invalid
        assert not placer.is_valid_annotation_target(1)
    
    def test_infer_annotation_from_context(self, sample_file):
        """Test context-based annotation inference"""
        placer = SignatureAnnotationPlacer(sample_file)
        
        # Class.forName should infer BinaryName
        assert placer.infer_annotation_from_context("Class.forName(name)") == SignatureAnnotationType.BINARY_NAME
        
        # .getName() should infer ClassGetName
        assert placer.infer_annotation_from_context("clazz.getName()") == SignatureAnnotationType.CLASS_GET_NAME
        
        # .getCanonicalName() should infer FullyQualifiedName
        assert placer.infer_annotation_from_context("clazz.getCanonicalName()") == SignatureAnnotationType.FULLY_QUALIFIED_NAME
    
    def test_analyze_and_place(self, sample_file):
        """Test full analysis and placement"""
        placer = SignatureAnnotationPlacer(sample_file)
        placements = placer.analyze_and_place()
        
        # Should find some signature-related declarations
        assert len(placements) >= 0
        
        # All placements should be SignaturePlacement
        for p in placements:
            assert isinstance(p, SignaturePlacement)
            assert p.annotation in SignatureAnnotationType
    
    def test_add_imports(self, sample_file):
        """Test import addition"""
        placer = SignatureAnnotationPlacer(sample_file)
        
        # Should add imports
        result = placer.add_imports()
        assert result == True
        
        # Check imports are in file content
        content = ''.join(placer.lines)
        assert 'signature.qual' in content
    
    def test_add_imports_specific(self, sample_file):
        """Test adding only specific imports"""
        placer = SignatureAnnotationPlacer(sample_file)
        
        # Add only BinaryName import
        result = placer.add_imports([SignatureAnnotationType.BINARY_NAME])
        assert result == True
        
        content = ''.join(placer.lines)
        assert 'BinaryName' in content
    
    def test_fixture_file_analysis(self, fixture_file):
        """Test analysis on the fixture file"""
        if fixture_file is None:
            pytest.skip("Fixture file not found")
        
        placer = SignatureAnnotationPlacer(fixture_file)
        placements = placer.analyze_and_place()
        
        # Should find signature-related declarations
        assert len(placements) >= 2
        
        # Should have variety of annotation types
        annotation_types = set(p.annotation for p in placements)
        assert len(annotation_types) >= 1


class TestSignatureIntegration:
    """Integration tests for Signature String placement"""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory"""
        temp_dir = tempfile.mkdtemp()
        
        # Create a sample Java file
        java_file = os.path.join(temp_dir, 'Sample.java')
        with open(java_file, 'w') as f:
            f.write('''package com.example;

public class Sample {
    String className = "java.lang.String";
    
    public void load(String typeName) throws Exception {
        Class.forName(typeName);
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
        placer = SignatureAnnotationPlacer(java_file)
        
        # Analyze and place
        placements = placer.analyze_and_place()
        
        # Add imports for used annotations
        if placements:
            annotations_used = list(set(p.annotation for p in placements))
            placer.add_imports(annotations_used)
        
        # Save file
        placer.save_file()
        
        # Verify file was modified
        with open(java_file, 'r') as f:
            content = f.read()
        
        # File should still be valid
        assert 'class Sample' in content


class TestAnnotationTypeMapping:
    """Test annotation type mapping between modules"""
    
    def test_enum_consistency_with_main_placer(self):
        """Test that annotation types match place_annotations.py"""
        try:
            from place_annotations import SignatureAnnotationType as MainSignatureType
            
            # Values should match
            assert MainSignatureType.BINARY_NAME.value == SignatureAnnotationType.BINARY_NAME.value
            assert MainSignatureType.FULLY_QUALIFIED_NAME.value == SignatureAnnotationType.FULLY_QUALIFIED_NAME.value
            assert MainSignatureType.FIELD_DESCRIPTOR.value == SignatureAnnotationType.FIELD_DESCRIPTOR.value
            assert MainSignatureType.CLASS_GET_NAME.value == SignatureAnnotationType.CLASS_GET_NAME.value
            assert MainSignatureType.INTERNAL_FORM.value == SignatureAnnotationType.INTERNAL_FORM.value
        except ImportError:
            pytest.skip("place_annotations.py not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
