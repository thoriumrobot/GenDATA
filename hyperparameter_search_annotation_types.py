#!/usr/bin/env python3
"""
Hyperparameter search for annotation type models.
Tests all 18 combinations (6 base models × 3 annotation types) with various hyperparameters.
"""

import subprocess
import json
import os
import logging
from datetime import datetime
import itertools

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_hyperparameter_search():
    """Run hyperparameter search for annotation type models"""
    
    # Define parameter grids for each model type
    param_grids = {
        'gcn': {
            'learning_rate': [0.001, 0.01, 0.1],
            'hidden_dim': [64, 128, 256],
            'dropout_rate': [0.1, 0.3, 0.5],
            'episodes': [20, 50, 100]
        },
        'gbt': {
            'learning_rate': [0.001, 0.01, 0.1],
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'episodes': [20, 50, 100]
        },
        'causal': {
            'learning_rate': [0.001, 0.01, 0.1],
            'hidden_dim': [64, 128, 256],
            'dropout_rate': [0.1, 0.3, 0.5],
            'episodes': [20, 50, 100]
        },
        'hgt': {
            'learning_rate': [0.001, 0.01, 0.1],
            'hidden_dim': [64, 128, 256],
            'dropout_rate': [0.1, 0.3, 0.5],
            'episodes': [20, 50, 100]
        },
        'gcsn': {
            'learning_rate': [0.001, 0.01, 0.1],
            'hidden_dim': [64, 128, 256],
            'dropout_rate': [0.1, 0.3, 0.5],
            'episodes': [20, 50, 100]
        },
        'dg2n': {
            'learning_rate': [0.001, 0.01, 0.1],
            'hidden_dim': [64, 128, 256],
            'dropout_rate': [0.1, 0.3, 0.5],
            'episodes': [20, 50, 100]
        }
    }
    
    # Annotation types
    annotation_types = ['positive', 'nonnegative', 'gtenegativeone']
    
    # Results storage
    results = {}
    
    # Limit combinations for testing (use first 3 combinations per model type)
    max_combinations = 3
    
    for annotation_type in annotation_types:
        results[annotation_type] = {}
        
        for base_model in param_grids.keys():
            logger.info(f"Starting hyperparameter search for {annotation_type}_{base_model}")
            
            param_grid = param_grids[base_model]
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            # Generate combinations
            combinations = list(itertools.product(*param_values))
            
            # Limit to max_combinations
            combinations = combinations[:max_combinations]
            
            model_results = []
            
            for i, combination in enumerate(combinations):
                logger.info(f"Testing combination {i+1}/{len(combinations)} for {annotation_type}_{base_model}")
                
                # Build command
                script_map = {
                    'positive': 'annotation_type_rl_positive.py',
                    'nonnegative': 'annotation_type_rl_nonnegative.py',
                    'gtenegativeone': 'annotation_type_rl_gtenegativeone.py'
                }
                
                cmd = [
                    'python', script_map[annotation_type],
                    '--base_model', base_model,
                    '--project_root', '/home/ubuntu/checker-framework/checker/tests/index'
                ]
                
                # Add hyperparameters
                params = dict(zip(param_names, combination))
                for param, value in params.items():
                    cmd.extend([f'--{param}', str(value)])
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
                    
                    if result.returncode == 0:
                        # Parse output for metrics
                        output = result.stdout
                        
                        # Extract final reward and predictions from output
                        final_reward = None
                        total_predictions = None
                        
                        for line in output.split('\n'):
                            if 'Episode completed:' in line and 'reward=' in line:
                                try:
                                    # Extract reward from line like "Episode completed: @Positive reward=0.246, predictions=3"
                                    reward_part = line.split('reward=')[1].split(',')[0]
                                    final_reward = float(reward_part)
                                except:
                                    pass
                            
                            if 'predictions=' in line:
                                try:
                                    # Extract predictions count
                                    pred_part = line.split('predictions=')[1].strip()
                                    total_predictions = int(pred_part)
                                except:
                                    pass
                        
                        # Calculate score (reward * predictions)
                        score = 0.0
                        if final_reward is not None and total_predictions is not None:
                            score = final_reward * total_predictions
                        
                        model_results.append({
                            'params': params,
                            'final_reward': final_reward,
                            'total_predictions': total_predictions,
                            'score': score,
                            'status': 'success'
                        })
                        
                        logger.info(f"Success: reward={final_reward}, predictions={total_predictions}, score={score}")
                        
                    else:
                        logger.error(f"Failed: {result.stderr}")
                        model_results.append({
                            'params': params,
                            'final_reward': None,
                            'total_predictions': None,
                            'score': 0.0,
                            'status': 'failed',
                            'error': result.stderr
                        })
                        
                except subprocess.TimeoutExpired:
                    logger.error(f"Timeout for combination {i+1}")
                    model_results.append({
                        'params': params,
                        'final_reward': None,
                        'total_predictions': None,
                        'score': 0.0,
                        'status': 'timeout'
                    })
                except Exception as e:
                    logger.error(f"Error for combination {i+1}: {e}")
                    model_results.append({
                        'params': params,
                        'final_reward': None,
                        'total_predictions': None,
                        'score': 0.0,
                        'status': 'error',
                        'error': str(e)
                    })
            
            # Find best parameters
            successful_results = [r for r in model_results if r['status'] == 'success' and r['score'] > 0]
            
            if successful_results:
                best_result = max(successful_results, key=lambda x: x['score'])
                results[annotation_type][base_model] = {
                    'best_params': best_result['params'],
                    'best_score': best_result['score'],
                    'best_reward': best_result['final_reward'],
                    'best_predictions': best_result['total_predictions'],
                    'all_results': model_results
                }
                logger.info(f"Best for {annotation_type}_{base_model}: score={best_result['score']}, params={best_result['params']}")
            else:
                results[annotation_type][base_model] = {
                    'best_params': None,
                    'best_score': 0.0,
                    'best_reward': None,
                    'best_predictions': None,
                    'all_results': model_results
                }
                logger.warning(f"No successful results for {annotation_type}_{base_model}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"hyperparameter_search_annotation_types_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Hyperparameter search completed. Results saved to {results_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("HYPERPARAMETER SEARCH SUMMARY")
    print("="*70)
    
    for annotation_type in annotation_types:
        print(f"\n{annotation_type.upper()} ANNOTATION TYPE:")
        for base_model in param_grids.keys():
            model_key = f"{annotation_type}_{base_model}"
            if model_key in results[annotation_type]:
                result = results[annotation_type][base_model]
                if result['best_params']:
                    print(f"  {base_model.upper()}: score={result['best_score']:.4f}, reward={result['best_reward']}, predictions={result['best_predictions']}")
                else:
                    print(f"  {base_model.upper()}: No successful results")
    
    return results

if __name__ == "__main__":
    results = run_hyperparameter_search()
