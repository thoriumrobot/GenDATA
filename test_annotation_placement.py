#!/usr/bin/env python3
"""
Integration tests for annotation placement system

Tests end-to-end flow: prediction generation and annotation placement.
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from place_annotations import ComprehensiveAnnotationPlacer, PredictionResult


class TestAnnotationPlacement(unittest.TestCase):
    """Integration tests for annotation placement"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / 'project'
        self.project_root.mkdir(parents=True)
        self.output_dir = Path(self.temp_dir) / 'output'
        
        # Create a test Java file
        self.test_file = self.project_root / 'Test.java'
        with open(self.test_file, 'w') as f:
            f.write("""public class Test {
    public void method(int count) {
        int result = count + 1;
        return result;
    }
}
""")
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_with_checker_name(self):
        """Test initialization with checker name"""
        placer = ComprehensiveAnnotationPlacer(
            project_root=str(self.project_root),
            output_dir=str(self.output_dir),
            checker_name='sql_quotes'
        )
        self.assertEqual(placer.checker_name, 'sql_quotes')
    
    def test_process_predictions_single_location(self):
        """Test processing predictions for a single location"""
        placer = ComprehensiveAnnotationPlacer(
            project_root=str(self.project_root),
            output_dir=str(self.output_dir),
            checker_name='lower_bound'
        )
        
        predictions = [
            PredictionResult(
                file_path='Test.java',
                line_number=3,
                confidence=0.8,
                annotation_type='@Positive',
                target_element='count',
                context='',
                model_type='causal'
            )
        ]
        
        with patch.object(placer, 'place_annotation_at_location', return_value=True):
            stats = placer.process_predictions(predictions)
            
            self.assertEqual(stats['total'], 1)
            self.assertEqual(stats['successful'], 1)
            self.assertEqual(stats['locations_with_predictions'], 1)
            self.assertEqual(stats['locations_after_selection'], 1)
    
    def test_process_predictions_multiple_same_location(self):
        """Test processing multiple predictions for same location - should select highest confidence"""
        placer = ComprehensiveAnnotationPlacer(
            project_root=str(self.project_root),
            output_dir=str(self.output_dir),
            checker_name='lower_bound'
        )
        
        predictions = [
            PredictionResult(
                file_path='Test.java',
                line_number=3,
                confidence=0.6,
                annotation_type='@Positive',
                target_element='count',
                context='',
                model_type='causal'
            ),
            PredictionResult(
                file_path='Test.java',
                line_number=3,
                confidence=0.9,  # Higher confidence
                annotation_type='@NonNegative',
                target_element='count',
                context='',
                model_type='gbt'
            ),
            PredictionResult(
                file_path='Test.java',
                line_number=3,
                confidence=0.7,
                annotation_type='@GTENegativeOne',
                target_element='count',
                context='',
                model_type='hgt'
            )
        ]
        
        placed_annotations = []
        
        def mock_place(file_path, line_num, annotations, strategy):
            placed_annotations.append((file_path, line_num, annotations))
            return True
        
        with patch.object(placer, 'place_annotation_at_location', side_effect=mock_place):
            stats = placer.process_predictions(predictions)
            
            # Should only place one annotation (highest confidence)
            self.assertEqual(len(placed_annotations), 1)
            self.assertEqual(placed_annotations[0][2], ['@NonNegative'])  # Highest confidence
            self.assertEqual(stats['locations_after_selection'], 1)
    
    def test_process_predictions_multiple_locations(self):
        """Test processing predictions for multiple locations"""
        placer = ComprehensiveAnnotationPlacer(
            project_root=str(self.project_root),
            output_dir=str(self.output_dir),
            checker_name='lower_bound'
        )
        
        predictions = [
            PredictionResult(
                file_path='Test.java',
                line_number=2,
                confidence=0.8,
                annotation_type='@Positive',
                target_element='count',
                context='',
                model_type='causal'
            ),
            PredictionResult(
                file_path='Test.java',
                line_number=3,
                confidence=0.7,
                annotation_type='@NonNegative',
                target_element='result',
                context='',
                model_type='gbt'
            )
        ]
        
        placed_annotations = []
        
        def mock_place(file_path, line_num, annotations, strategy):
            placed_annotations.append((file_path, line_num, annotations))
            return True
        
        with patch.object(placer, 'place_annotation_at_location', side_effect=mock_place):
            stats = placer.process_predictions(predictions)
            
            # Should place annotations at both locations
            self.assertEqual(len(placed_annotations), 2)
            self.assertEqual(stats['locations_after_selection'], 2)
    
    def test_process_predictions_verifies_single_annotation(self):
        """Test that only one annotation is placed per location"""
        placer = ComprehensiveAnnotationPlacer(
            project_root=str(self.project_root),
            output_dir=str(self.output_dir),
            checker_name='lower_bound'
        )
        
        predictions = [
            PredictionResult(
                file_path='Test.java',
                line_number=3,
                confidence=0.8,
                annotation_type='@Positive',
                target_element='count',
                context='',
                model_type='causal'
            )
        ]
        
        def mock_place(file_path, line_num, annotations, strategy):
            # Simulate multiple annotations being passed (should not happen)
            if len(annotations) > 1:
                raise AssertionError(f"Multiple annotations passed: {annotations}")
            return True
        
        with patch.object(placer, 'place_annotation_at_location', side_effect=mock_place):
            stats = placer.process_predictions(predictions)
            self.assertEqual(stats['successful'], 1)
    
    def test_select_appropriate_annotation_returns_single(self):
        """Test that select_appropriate_annotation returns only one annotation"""
        placer = ComprehensiveAnnotationPlacer(
            project_root=str(self.project_root),
            output_dir=str(self.output_dir),
            checker_name='lower_bound'
        )
        
        prediction = PredictionResult(
            file_path='Test.java',
            line_number=3,
            confidence=0.8,
            annotation_type='@Positive',
            target_element='count',
            context='',
            model_type='causal'
        )
        
        context = Mock()
        context.code_line = 'int count = 0'
        
        annotations = placer.select_appropriate_annotation(prediction, context)
        
        # Should return exactly one annotation
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0], '@Positive')
    
    def test_load_predictions_json_list(self):
        """Test loading predictions from JSON list format"""
        placer = ComprehensiveAnnotationPlacer(
            project_root=str(self.project_root),
            output_dir=str(self.output_dir),
            checker_name='lower_bound'
        )
        
        predictions_file = Path(self.temp_dir) / 'predictions.json'
        predictions_data = [
            {
                'file_path': 'Test.java',
                'line_number': 3,
                'confidence': 0.8,
                'annotation_type': '@Positive',
                'target_element': 'count',
                'model_type': 'causal'
            }
        ]
        
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f)
        
        predictions = placer.load_predictions(str(predictions_file))
        
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].annotation_type, '@Positive')
        self.assertEqual(predictions[0].line_number, 3)
    
    def test_load_predictions_json_dict(self):
        """Test loading predictions from JSON dict format"""
        placer = ComprehensiveAnnotationPlacer(
            project_root=str(self.project_root),
            output_dir=str(self.output_dir),
            checker_name='lower_bound'
        )
        
        predictions_file = Path(self.temp_dir) / 'predictions.json'
        predictions_data = {
            'Test.java': [
                {
                    'line_number': 3,
                    'confidence': 0.8,
                    'annotation_type': '@Positive',
                    'target_element': 'count',
                    'model_type': 'causal'
                }
            ]
        }
        
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f)
        
        predictions = placer.load_predictions(str(predictions_file))
        
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].file_path, 'Test.java')


class TestConfidenceSelection(unittest.TestCase):
    """Test confidence-based selection logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / 'project'
        self.project_root.mkdir(parents=True)
        self.output_dir = Path(self.temp_dir) / 'output'
        
        # Create a test Java file
        self.test_file = self.project_root / 'Test.java'
        with open(self.test_file, 'w') as f:
            f.write("""public class Test {
    public void method(int count) {
        int result = count + 1;
        return result;
    }
}
""")
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_equal_confidence_selects_deterministically(self):
        """Test that equal confidence scores are handled deterministically"""
        placer = ComprehensiveAnnotationPlacer(
            project_root=str(self.project_root),
            output_dir=str(self.output_dir),
            checker_name='lower_bound'
        )
        
        # Use absolute path to test file
        test_file_path = str(self.test_file)
        
        predictions = [
            PredictionResult(
                file_path=test_file_path,  # Use absolute path
                line_number=3,
                confidence=0.8,
                annotation_type='@Positive',
                target_element='count',
                context='',
                model_type='causal'
            ),
            PredictionResult(
                file_path=test_file_path,  # Use absolute path
                line_number=3,
                confidence=0.8,  # Same confidence
                annotation_type='@NonNegative',
                target_element='count',
                context='',
                model_type='gbt'
            )
        ]
        
        placed_annotations = []
        
        def mock_place(file_path, line_num, annotations, strategy):
            placed_annotations.append((file_path, line_num, annotations))
            return True
        
        with patch.object(placer, 'place_annotation_at_location', side_effect=mock_place):
            stats = placer.process_predictions(predictions)
            
            # Should select one deterministically (first one found by max())
            self.assertEqual(len(placed_annotations), 1)
            self.assertEqual(stats['locations_after_selection'], 1)


if __name__ == '__main__':
    unittest.main()

