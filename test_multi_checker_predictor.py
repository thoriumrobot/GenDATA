#!/usr/bin/env python3
"""
Unit tests for MultiCheckerPredictor

Tests model loading, feature extraction, prediction logic, and file prediction.
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn

from multi_checker_predictor import MultiCheckerPredictor


class TestMultiCheckerPredictor(unittest.TestCase):
    """Unit tests for MultiCheckerPredictor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.models_dir = self.temp_dir / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a proper model matching ImprovedBalancedAnnotationTypeModel structure
        from improved_balanced_annotation_type_trainer import ImprovedBalancedAnnotationTypeModel
        self.mock_model = ImprovedBalancedAnnotationTypeModel(
            input_dim=21,
            hidden_dims=[512, 256, 128, 64],
            dropout_rate=0.4
        )
        
        self.mock_checkpoint = {
            'model_state_dict': self.mock_model.state_dict(),
            'input_dim': 21,
            'model_type': 'improved_balanced_causal',
            'training_stats': {}
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_lower_bound(self):
        """Test initialization for Lower Bound checker"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        self.assertEqual(predictor.checker_name, 'lower_bound')
        self.assertEqual(len(predictor.annotation_types), 3)
        self.assertIn('@Positive', predictor.annotation_types)
        self.assertIn('@NonNegative', predictor.annotation_types)
        self.assertIn('@GTENegativeOne', predictor.annotation_types)
    
    def test_init_sql_quotes(self):
        """Test initialization for SQL Quotes checker"""
        predictor = MultiCheckerPredictor('sql_quotes', models_dir=str(self.models_dir))
        self.assertEqual(predictor.checker_name, 'sql_quotes')
        self.assertEqual(len(predictor.annotation_types), 2)
        self.assertIn('@SqlEvenQuotes', predictor.annotation_types)
        self.assertIn('@SqlOddQuotes', predictor.annotation_types)
    
    def test_init_signature_string(self):
        """Test initialization for Signature String checker"""
        predictor = MultiCheckerPredictor('signature_string', models_dir=str(self.models_dir))
        self.assertEqual(predictor.checker_name, 'signature_string')
        self.assertEqual(len(predictor.annotation_types), 3)
        self.assertIn('@FullyQualifiedName', predictor.annotation_types)
        self.assertIn('@BinaryName', predictor.annotation_types)
        self.assertIn('@FieldDescriptor', predictor.annotation_types)
    
    def test_init_unknown_checker(self):
        """Test initialization with unknown checker raises error"""
        with self.assertRaises(ValueError):
            MultiCheckerPredictor('unknown_checker', models_dir=str(self.models_dir))
    
    def test_load_checker_models_no_models(self):
        """Test loading when no models exist"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        result = predictor.load_checker_models()
        self.assertFalse(result)
        # loaded_models has empty dicts for each annotation type, but no actual models loaded
        total_loaded = sum(len(models) for models in predictor.loaded_models.values())
        self.assertEqual(total_loaded, 0)
    
    def test_load_checker_models_success(self):
        """Test successful model loading"""
        # Create a mock model file
        model_file = self.models_dir / 'positive_causal_balanced_model.pth'
        torch.save(self.mock_checkpoint, model_file)
        
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        result = predictor.load_checker_models()
        
        # Should load at least one model
        self.assertTrue(result)
        self.assertGreater(len(predictor.loaded_models), 0)
    
    def test_load_checker_models_lower_bound_legacy(self):
        """Test loading Lower Bound models without _balanced suffix"""
        # Create legacy model file
        model_file = self.models_dir / 'positive_causal_model.pth'
        torch.save(self.mock_checkpoint, model_file)
        
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        result = predictor.load_checker_models()
        
        # Should find and load the legacy model
        self.assertTrue(result)
    
    def test_extract_features(self):
        """Test feature extraction"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        
        node = {
            'id': 0,
            'label': 'int count = 0',
            'line': 10,
            'node_type': 'statement'
        }
        
        cfg_data = {
            'nodes': [node],
            'edges': []
        }
        
        with patch('improved_balanced_dataset_generator.ImprovedBalancedDatasetGenerator') as mock_gen_class:
            mock_generator = Mock()
            mock_generator.extract_node_features.return_value = [1.0] * 21
            mock_gen_class.return_value = mock_generator
            
            features = predictor._extract_features(node, cfg_data, 'lower_bound')
            self.assertEqual(len(features), 21)
            self.assertTrue(all(isinstance(f, (int, float)) for f in features))
    
    def test_extract_features_error(self):
        """Test feature extraction with error"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        
        node = {'id': 0, 'label': 'test', 'line': 10}
        cfg_data = {'nodes': [node], 'edges': []}
        
        with patch('improved_balanced_dataset_generator.ImprovedBalancedDatasetGenerator', side_effect=Exception("Test error")):
            features = predictor._extract_features(node, cfg_data, 'lower_bound')
            self.assertEqual(features, [])
    
    def test_get_model_prediction_positive(self):
        """Test model prediction with positive result"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        
        # Create a mock model that returns positive prediction
        mock_model = Mock()
        mock_output = torch.tensor([[0.2, 0.8]])  # Class 1 (positive) with high probability
        mock_model.return_value = mock_output
        
        model_info = {
            'model': mock_model,
            'type': 'feature_based',
            'input_dim': 21
        }
        
        features = [1.0] * 21
        
        # Mock the model to return the expected output
        mock_model.return_value = mock_output
        
        with patch('torch.no_grad'), patch('torch.softmax', return_value=torch.tensor([[0.2, 0.8]])):
            with patch('torch.argmax', return_value=torch.tensor([1])):
                is_positive, confidence, reason = predictor._get_model_prediction(
                    model_info, features, '@Positive', 'causal'
                )
                self.assertTrue(is_positive)
                self.assertGreater(confidence, 0.0)
                self.assertIn('@Positive', reason)
    
    def test_get_model_prediction_negative(self):
        """Test model prediction with negative result"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        
        mock_model = Mock()
        mock_output = torch.tensor([[0.8, 0.2]])  # Class 0 (negative)
        mock_model.return_value = mock_output
        
        model_info = {
            'model': mock_model,
            'type': 'feature_based',
            'input_dim': 21
        }
        
        features = [1.0] * 21
        
        mock_model.return_value = mock_output
        
        with patch('torch.no_grad'), patch('torch.softmax', return_value=torch.tensor([[0.8, 0.2]])):
            with patch('torch.argmax', return_value=torch.tensor([0])):
                is_positive, confidence, reason = predictor._get_model_prediction(
                    model_info, features, '@Positive', 'causal'
                )
                self.assertFalse(is_positive)
    
    def test_predict_for_location_no_models(self):
        """Test prediction when no models are loaded"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        
        node = {'id': 0, 'label': 'test', 'line': 10}
        cfg_data = {'nodes': [node], 'edges': []}
        
        result = predictor.predict_for_location(cfg_data, node, 10)
        self.assertIsNone(result)
    
    def test_predict_for_location_no_positive(self):
        """Test prediction when no models predict positive"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        
        # Mock loaded models that all return negative
        predictor.loaded_models = {
            '@Positive': {
                'causal': {
                    'model': Mock(),
                    'type': 'feature_based',
                    'input_dim': 21
                }
            }
        }
        
        node = {'id': 0, 'label': 'test', 'line': 10}
        cfg_data = {'nodes': [node], 'edges': []}
        
        with patch.object(predictor, '_extract_features', return_value=[1.0] * 21):
            with patch.object(predictor, '_get_model_prediction', return_value=(False, 0.5, 'No annotation')):
                result = predictor.predict_for_location(cfg_data, node, 10)
                self.assertIsNone(result)
    
    def test_predict_for_location_selects_highest_confidence(self):
        """Test that highest confidence prediction is selected"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        
        predictor.loaded_models = {
            '@Positive': {
                'causal': {'model': Mock(), 'type': 'feature_based', 'input_dim': 21},
                'gbt': {'model': Mock(), 'type': 'feature_based', 'input_dim': 21}
            },
            '@NonNegative': {
                'causal': {'model': Mock(), 'type': 'feature_based', 'input_dim': 21}
            }
        }
        
        node = {'id': 0, 'label': 'test', 'line': 10}
        cfg_data = {'nodes': [node], 'edges': []}
        
        # Mock predictions with different confidences
        predictions = [
            ('@Positive', 'causal', True, 0.6, 'reason1'),
            ('@Positive', 'gbt', True, 0.9, 'reason2'),  # Highest confidence
            ('@NonNegative', 'causal', True, 0.7, 'reason3')
        ]
        
        with patch.object(predictor, '_extract_features', return_value=[1.0] * 21):
            with patch.object(predictor, '_get_model_prediction', side_effect=lambda mi, f, at, bm: {
                ('@Positive', 'causal'): (True, 0.6, 'reason1'),
                ('@Positive', 'gbt'): (True, 0.9, 'reason2'),
                ('@NonNegative', 'causal'): (True, 0.7, 'reason3')
            }[(at, bm)]):
                result = predictor.predict_for_location(cfg_data, node, 10, threshold=0.3)
                
                self.assertIsNotNone(result)
                self.assertEqual(result['annotation_type'], '@Positive')
                self.assertEqual(result['model_type'], 'gbt')
                self.assertEqual(result['confidence'], 0.9)
    
    def test_predict_for_file_missing_cfg(self):
        """Test prediction with missing CFG file"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        predictor.loaded_models = {'@Positive': {'causal': {'model': Mock(), 'type': 'feature_based', 'input_dim': 21}}}
        
        java_file = '/nonexistent/file.java'
        cfg_dir = str(self.temp_dir / 'cfg')
        Path(cfg_dir).mkdir(exist_ok=True)
        
        result = predictor.predict_for_file(java_file, cfg_dir)
        self.assertEqual(result, [])
    
    def test_predict_for_file_valid_cfg(self):
        """Test prediction with valid CFG file"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        
        # Create CFG file
        cfg_dir = Path(self.temp_dir) / 'cfg'
        cfg_dir.mkdir()
        cfg_file = cfg_dir / 'Test.java.json'
        
        cfg_data = {
            'nodes': [
                {'id': 0, 'label': 'int count = 0', 'line': 10, 'node_type': 'statement'},
                {'id': 1, 'label': 'count++', 'line': 11, 'node_type': 'statement'}
            ],
            'edges': []
        }
        
        with open(cfg_file, 'w') as f:
            json.dump(cfg_data, f)
        
        java_file = '/path/to/Test.java'
        
        # Mock model loading and prediction
        predictor.loaded_models = {
            '@Positive': {
                'causal': {'model': Mock(), 'type': 'feature_based', 'input_dim': 21}
            }
        }
        
        # Mock _find_cfg_file to return the correct path
        with patch.object(predictor, '_find_cfg_file', return_value=str(cfg_file)):
            # Mock predict_for_location to return prediction for each node
            def mock_predict(cfg_data, node, line_num, threshold):
                if line_num == 10:
                    return {
                        'annotation_type': '@Positive',
                        'confidence': 0.8,
                        'model_type': 'causal',
                        'reason': 'test',
                        'line_number': 10
                    }
                elif line_num == 11:
                    return {
                        'annotation_type': '@Positive',
                        'confidence': 0.7,
                        'model_type': 'causal',
                        'reason': 'test',
                        'line_number': 11
                    }
                return None
            
            with patch.object(predictor, 'predict_for_location', side_effect=mock_predict):
                result = predictor.predict_for_file(java_file, str(cfg_dir))
                self.assertEqual(len(result), 2)  # One per node
    
    def test_find_cfg_file_direct_match(self):
        """Test finding CFG file with direct match"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        
        cfg_dir = Path(self.temp_dir) / 'cfg'
        cfg_dir.mkdir()
        cfg_file = cfg_dir / 'Test.json'
        cfg_file.touch()
        
        java_file = '/path/to/Test.java'
        result = predictor._find_cfg_file(java_file, str(cfg_dir))
        
        self.assertEqual(result, str(cfg_file))
    
    def test_find_cfg_file_subdirectory(self):
        """Test finding CFG file in subdirectory"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        
        cfg_dir = Path(self.temp_dir) / 'cfg'
        cfg_dir.mkdir()
        subdir = cfg_dir / 'Test'
        subdir.mkdir()
        cfg_file = subdir / 'cfg.json'
        cfg_file.touch()
        
        java_file = '/path/to/Test.java'
        result = predictor._find_cfg_file(java_file, str(cfg_dir))
        
        self.assertEqual(result, str(cfg_file))
    
    def test_find_cfg_file_not_found(self):
        """Test finding CFG file when not found"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        
        cfg_dir = Path(self.temp_dir) / 'cfg'
        cfg_dir.mkdir()
        
        java_file = '/path/to/Nonexistent.java'
        result = predictor._find_cfg_file(java_file, str(cfg_dir))
        
        self.assertIsNone(result)


class TestCFGValidation(unittest.TestCase):
    """Test CFG validation and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_predict_for_file_invalid_json(self):
        """Test handling of invalid JSON in CFG file"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        
        cfg_dir = Path(self.temp_dir) / 'cfg'
        cfg_dir.mkdir()
        cfg_file = cfg_dir / 'Test.json'
        
        # Write invalid JSON
        with open(cfg_file, 'w') as f:
            f.write('invalid json {')
        
        java_file = '/path/to/Test.java'
        result = predictor.predict_for_file(java_file, str(cfg_dir))
        
        self.assertEqual(result, [])
    
    def test_predict_for_file_missing_nodes(self):
        """Test handling of CFG file missing nodes field"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        
        cfg_dir = Path(self.temp_dir) / 'cfg'
        cfg_dir.mkdir()
        cfg_file = cfg_dir / 'Test.json'
        
        cfg_data = {'edges': []}  # Missing 'nodes'
        
        with open(cfg_file, 'w') as f:
            json.dump(cfg_data, f)
        
        java_file = '/path/to/Test.java'
        result = predictor.predict_for_file(java_file, str(cfg_dir))
        
        self.assertEqual(result, [])
    
    def test_predict_for_file_empty_nodes(self):
        """Test handling of CFG file with empty nodes"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        
        cfg_dir = Path(self.temp_dir) / 'cfg'
        cfg_dir.mkdir()
        cfg_file = cfg_dir / 'Test.json'
        
        cfg_data = {'nodes': [], 'edges': []}
        
        with open(cfg_file, 'w') as f:
            json.dump(cfg_data, f)
        
        java_file = '/path/to/Test.java'
        result = predictor.predict_for_file(java_file, str(cfg_dir))
        
        self.assertEqual(result, [])
    
    def test_predict_for_file_nodes_without_line(self):
        """Test handling of nodes without line numbers"""
        predictor = MultiCheckerPredictor('lower_bound', models_dir=str(self.models_dir))
        
        cfg_dir = Path(self.temp_dir) / 'cfg'
        cfg_dir.mkdir()
        cfg_file = cfg_dir / 'Test.json'
        
        cfg_data = {
            'nodes': [
                {'id': 0, 'label': 'test', 'node_type': 'statement'}  # No 'line' or 'line_number'
            ],
            'edges': []
        }
        
        with open(cfg_file, 'w') as f:
            json.dump(cfg_data, f)
        
        java_file = '/path/to/Test.java'
        predictor.loaded_models = {'@Positive': {'causal': {'model': Mock(), 'type': 'feature_based', 'input_dim': 21}}}
        
        result = predictor.predict_for_file(java_file, str(cfg_dir))
        
        # Should skip nodes without line numbers
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()

