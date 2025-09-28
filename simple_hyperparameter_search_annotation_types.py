#!/usr/bin/env python3
"""
Simple hyperparameter search for annotation type models.
Tests all 18 combinations (6 base models × 3 annotation types) with different episode counts.
"""

import subprocess
import json
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_simple_hyperparameter_search():
    """Run simple hyperparameter search for annotation type models"""
    
    # Test different episode counts
    episode_counts = [10, 25, 50]
    
    # Annotation types and base models
    annotation_types = ['positive', 'nonnegative', 'gtenegativeone']
    base_models = ['gcn', 'gbt', 'causal', 'hgt', 'gcsn', 'dg2n']
    
    # Results storage
    results = {}
    
    for annotation_type in annotation_types:
        results[annotation_type] = {}
        
        for base_model in base_models:
            logger.info(f"Testing {annotation_type}_{base_model} with different episode counts")
            
            model_results = []
            
            for episodes in episode_counts:
                logger.info(f"Testing {annotation_type}_{base_model} with {episodes} episodes")
                
                # Build command
                script_map = {
                    'positive': 'annotation_type_rl_positive.py',
                    'nonnegative': 'annotation_type_rl_nonnegative.py',
                    'gtenegativeone': 'annotation_type_rl_gtenegativeone.py'
                }
                
                cmd = [
                    'python', script_map[annotation_type],
                    '--base_model', base_model,
                    '--episodes', str(episodes),
                    '--project_root', '/home/ubuntu/checker-framework/checker/tests/index'
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)  # 2 minute timeout
                    
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
                            'episodes': episodes,
                            'final_reward': final_reward,
                            'total_predictions': total_predictions,
                            'score': score,
                            'status': 'success'
                        })
                        
                        logger.info(f"Success: episodes={episodes}, reward={final_reward}, predictions={total_predictions}, score={score}")
                        
                    else:
                        logger.error(f"Failed with {episodes} episodes: {result.stderr}")
                        model_results.append({
                            'episodes': episodes,
                            'final_reward': None,
                            'total_predictions': None,
                            'score': 0.0,
                            'status': 'failed',
                            'error': result.stderr
                        })
                        
                except subprocess.TimeoutExpired:
                    logger.error(f"Timeout for {episodes} episodes")
                    model_results.append({
                        'episodes': episodes,
                        'final_reward': None,
                        'total_predictions': None,
                        'score': 0.0,
                        'status': 'timeout'
                    })
                except Exception as e:
                    logger.error(f"Error for {episodes} episodes: {e}")
                    model_results.append({
                        'episodes': episodes,
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
                    'best_episodes': best_result['episodes'],
                    'best_score': best_result['score'],
                    'best_reward': best_result['final_reward'],
                    'best_predictions': best_result['total_predictions'],
                    'all_results': model_results
                }
                logger.info(f"Best for {annotation_type}_{base_model}: episodes={best_result['episodes']}, score={best_result['score']}")
            else:
                results[annotation_type][base_model] = {
                    'best_episodes': None,
                    'best_score': 0.0,
                    'best_reward': None,
                    'best_predictions': None,
                    'all_results': model_results
                }
                logger.warning(f"No successful results for {annotation_type}_{base_model}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"simple_hyperparameter_search_annotation_types_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Simple hyperparameter search completed. Results saved to {results_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("SIMPLE HYPERPARAMETER SEARCH SUMMARY")
    print("="*70)
    
    for annotation_type in annotation_types:
        print(f"\n{annotation_type.upper()} ANNOTATION TYPE:")
        for base_model in base_models:
            model_key = f"{annotation_type}_{base_model}"
            if model_key in results[annotation_type]:
                result = results[annotation_type][base_model]
                if result['best_episodes']:
                    print(f"  {base_model.upper()}: episodes={result['best_episodes']}, score={result['best_score']:.4f}, reward={result['best_reward']}, predictions={result['best_predictions']}")
                else:
                    print(f"  {base_model.upper()}: No successful results")
    
    return results

if __name__ == "__main__":
    results = run_simple_hyperparameter_search()
