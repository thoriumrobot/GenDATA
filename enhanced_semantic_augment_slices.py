#!/usr/bin/env python3
"""
Enhanced Semantic-Preserving Data Augmentation Script

This script uses JDT-based semantic transformations instead of regex-based ones.
It provides enhanced transformations that increase code variety while preserving semantics
and being resistant to slicer pruning.

JDT-Based Transformations:
1. Method extraction and inlining
2. Conditional expression restructuring
3. Array access pattern variations
4. String concatenation alternatives
5. Numeric literal transformations
6. Exception handling restructuring
7. Lambda expression conversions
8. Stream API alternatives
9. Builder pattern variations
10. Functional programming conversions
"""

import os
import argparse
import random
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import time

from jdt_semantic_transformer import JdtSemanticTransformer

# Set up logging
logger = logging.getLogger(__name__)

HEADER_COMMENT = """
/*
 * CFWR enhanced semantic augmentation: applied advanced semantic-preserving transformations using JDT AST parsing.
 */
""".strip()

class EnhancedSemanticTransformer:
    """Enhanced semantic transformer using JDT-based transformations."""
    
    def __init__(self, seed: int = 42, disabled_transformations: List[str] = None, jdt_jar_path: Optional[str] = None):
        random.seed(seed)
        self.seed = seed
        self.transformations_applied = []
        self.disabled_transformations = disabled_transformations or []
        # Profiling store: list of timing entries per transformation attempt
        self._timing_records: List[Dict[str, Any]] = []
        
        # Initialize JDT transformer
        try:
            self.jdt_transformer = JdtSemanticTransformer(jdt_jar_path, seed)
            logger.info("Initialized EnhancedSemanticTransformer with JDT-based transformations")
        except Exception as e:
            logger.error(f"Failed to initialize JDT transformer: {e}")
            raise RuntimeError(f"JDT transformer initialization failed: {e}")
    
    def transform_file(self, java_path: str, variant_idx: int,
                       sequence_len: int = 2,
                       max_depth: int = 3,
                       avoid: Optional[List[str]] = None,
                       min_diff: float = 0.03,
                       focus_nodes: Optional[List[str]] = None,
                       max_retries: int = 2) -> str:
        """Apply enhanced semantic transformations to a Java file using JDT."""
        try:
            with open(java_path, 'r') as f:
                original_code = f.read()
            
            # Get available enhanced transformations
            available_transformations = self.jdt_transformer.get_available_transformations('enhanced')
            avoid = avoid or []
            focus_nodes = focus_nodes or ['control','dataflow']
            
            # Filter out disabled and avoided transformations
            enabled_transformations = [t for t in available_transformations
                                       if t not in self.disabled_transformations and t not in avoid]
            
            if not enabled_transformations:
                logger.warning("No transformations available after filtering disabled ones")
                return original_code
            
            # Use deterministic selection based on variant index to ensure different transformations
            # Set seed based on variant index for reproducible but different transformations
            variant_random = random.Random(self.seed + variant_idx * 1000)
            
            # Build a diverse pool and deterministically sample to increase variety across variants
            diversity_pool = [
                'variable_operation','ternary_operator','mathematical_expression','loop_conversion',
                'guard_reversal','switch_statement','logical_expression','conditional_expression',
                'stream_api','builder_pattern','functional_conversion','simple_field_access',
                'string_concatenation','numeric_literal'
            ]
            diversity_pool = [t for t in diversity_pool if t in enabled_transformations]
            if not diversity_pool:
                diversity_pool = enabled_transformations
            num_transformations = max(1, min(sequence_len, len(diversity_pool)))
            # Deterministic but varied per variant
            selected_transformations = variant_random.sample(diversity_pool, num_transformations)
            
            logger.info(f"Applying transformations for variant {variant_idx}: {selected_transformations}")
            
            # Apply transformations using JDT with forced transformation and bounded retries
            t0 = time.perf_counter()
            transformed_code = self.jdt_transformer.transform_code(
                original_code,
                selected_transformations,
                'enhanced',
                force_transformation=True,
                max_retries=max_retries,
                sequence_len=sequence_len,
                max_depth=max_depth,
                avoid=avoid,
                focus_nodes=focus_nodes
            )
            t1 = time.perf_counter()
            self._timing_records.append({
                'java_file': java_path,
                'variant_idx': variant_idx,
                'mode': 'enhanced',
                'stage': 'initial',
                'transformations': list(selected_transformations),
                'duration_ms': (t1 - t0) * 1000.0
            })
            used_transformations = list(selected_transformations)

            # Log body hash and size deltas
            try:
                import hashlib
                ob = original_code.encode('utf-8')
                nb = transformed_code.encode('utf-8')
                o_hash = hashlib.md5(ob).hexdigest()
                n_hash = hashlib.md5(nb).hexdigest()
                logger.debug(f"variant {variant_idx} body hash: {o_hash} -> {n_hash}, len: {len(ob)} -> {len(nb)}")
            except Exception:
                pass

            # Retry with alternates if unchanged
            if transformed_code == original_code:
                # Build diverse fallback candidates
                fallback_sets = [
                    ['loop_conversion','guard_reversal','mathematical_expression'],
                    ['logical_expression','conditional_expression'],
                    ['stream_api'],
                    ['builder_pattern'],
                    ['functional_conversion'],
                    ['string_concatenation'],
                    ['numeric_literal'],
                    ['simple_field_access'],
                    ['variable_operation'],
                    ['ternary_operator'],
                ]
                fallback_count = 0
                for alt in fallback_sets:
                    if fallback_count >= max_retries:
                        logger.debug(f"Max fallback attempts ({max_retries}) reached for variant {variant_idx}")
                        break
                    if not alt:
                        continue
                    t_alt0 = time.perf_counter()
                    alt_code = self.jdt_transformer.transform_code(
                        original_code, alt, 'enhanced', force_transformation=False, max_retries=0,
                        sequence_len=sequence_len, max_depth=max_depth, avoid=avoid, focus_nodes=focus_nodes)
                    t_alt1 = time.perf_counter()
                    self._timing_records.append({
                        'java_file': java_path,
                        'variant_idx': variant_idx,
                        'mode': 'enhanced',
                        'stage': 'fallback',
                        'transformations': list(alt),
                        'duration_ms': (t_alt1 - t_alt0) * 1000.0
                    })
                    if alt_code != original_code:
                        logger.info(f"Fallback transformations succeeded for variant {variant_idx}: {alt}")
                        transformed_code = alt_code
                        # Prefer applied list from wrapper if available
                        applied = getattr(self.jdt_transformer, '_last_applied', None)
                        used_transformations = list(applied) if applied else list(alt)
                        break
                    fallback_count += 1

            # Enforce minimal diff threshold (approx by token-level ratio)
            try:
                import re
                def tokenize(s: str) -> List[str]:
                    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\S", s)
                o_tokens = tokenize(original_code)
                n_tokens = tokenize(transformed_code)
                # simple diff: proportion of tokens that differ by position, clipped by min length
                L = min(len(o_tokens), len(n_tokens))
                diffs = sum(1 for i in range(L) if o_tokens[i] != n_tokens[i]) + abs(len(o_tokens) - len(n_tokens))
                diff_ratio = (diffs / max(1, max(len(o_tokens), len(n_tokens))))
                if diff_ratio < min_diff:
                    logger.debug(f"Variant {variant_idx} diff_ratio {diff_ratio:.3f} < min_diff {min_diff:.3f}; keeping original")
                    transformed_code = original_code
            except Exception:
                pass
            
            # Record applied transformations (selected or fallback used, or parsed from JAR)
            applied = getattr(self.jdt_transformer, '_last_applied', None)
            self.transformations_applied.extend(list(applied) if applied else used_transformations)
            
            # Always add header comment to indicate transformation attempt
            if transformed_code != original_code:
                transformed_code = self._add_header_comment(transformed_code, selected_transformations)
            else:
                # Even if no changes, add header to show transformation was attempted
                transformed_code = self._add_header_comment(transformed_code, [f"attempted_{t}" for t in selected_transformations])
            
            return transformed_code
            
        except Exception as e:
            logger.error(f"Failed to transform file {java_path}: {e}")
            return original_code
    
    def _add_header_comment(self, code: str, transformations: List[str]) -> str:
        """Add header comment with applied transformations."""
        transformation_list = ", ".join(transformations)
        header = f"{HEADER_COMMENT}\n// Applied transformations: {transformation_list}\n\n"
        return header + code
    
    def transform_directory(self, input_dir: str, output_dir: str, num_variants: int = 3,
                            sequence_len: int = 2, max_depth: int = 3, avoid: Optional[List[str]] = None,
                            min_diff: float = 0.03, focus_nodes: Optional[List[str]] = None,
                            write_manifest: bool = True, max_retries: int = 2) -> Dict[str, Any]:
        """Transform all Java files in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'files_processed': 0,
            'variants_created': 0,
            'transformations_applied': [],
            'errors': []
        }
        
        java_files = list(input_path.glob('**/*.java'))
        logger.info(f"Found {len(java_files)} Java files to process")
        
        for java_file in java_files:
            try:
                relative_path = java_file.relative_to(input_path)
                
                # Create variants
                for variant_idx in range(num_variants):
                    transformed_code = self.transform_file(
                        str(java_file), variant_idx,
                        sequence_len=sequence_len, max_depth=max_depth, avoid=avoid,
                        min_diff=min_diff, focus_nodes=focus_nodes, max_retries=max_retries)
                    
                    # Create output path
                    variant_name = f"{java_file.stem}_variant_{variant_idx}{java_file.suffix}"
                    output_file = output_path / relative_path.parent / variant_name
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write transformed file
                    with open(output_file, 'w') as f:
                        f.write(transformed_code)

                    # Write per-variant manifest
                    if write_manifest:
                        manifest = {
                            'source': str(java_file),
                            'output': str(output_file),
                            'seed': self.seed,
                            'variant_index': variant_idx,
                            'sequence_len': sequence_len,
                            'max_depth': max_depth,
                            'avoid': avoid or [],
                            'min_diff': min_diff,
                            'focus_nodes': focus_nodes or ['control','dataflow']
                        }
                        try:
                            import json
                            with open(str(output_file) + '.manifest.json', 'w') as mf:
                                json.dump(manifest, mf, indent=2)
                        except Exception:
                            pass
                    
                    results['variants_created'] += 1
                    logger.info(f"Created variant: {output_file}")
                
                results['files_processed'] += 1
                results['transformations_applied'].extend(self.transformations_applied)
                
            except Exception as e:
                error_msg = f"Error processing {java_file}: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        # Write timing report alongside index for later analysis
        try:
            import json
            timing_path = Path(output_dir) / 'augmentation_timing_report.json'
            # Also provide simple aggregates by transformation set label
            aggregates: Dict[str, Dict[str, Any]] = {}
            for rec in self._timing_records:
                label = ",".join(rec.get('transformations', [])) or '(none)'
                agg = aggregates.setdefault(label, {'count': 0, 'total_ms': 0.0, 'max_ms': 0.0})
                agg['count'] += 1
                agg['total_ms'] += rec['duration_ms']
                if rec['duration_ms'] > agg['max_ms']:
                    agg['max_ms'] = rec['duration_ms']
            for k, v in aggregates.items():
                v['avg_ms'] = (v['total_ms'] / v['count']) if v['count'] else 0.0
            with open(timing_path, 'w') as tf:
                json.dump({'records': self._timing_records, 'aggregates_by_set': aggregates}, tf, indent=2)
        except Exception:
            pass

        # Write index
        if write_manifest:
            try:
                import json
                index_path = Path(output_dir) / 'augmentation_index.json'
                with open(index_path, 'w') as idx:
                    json.dump(results, idx, indent=2)
            except Exception:
                pass
        return results
    
    def get_transformation_stats(self) -> Dict[str, Any]:
        """Get statistics about applied transformations."""
        transformation_counts = {}
        for transformation in self.transformations_applied:
            transformation_counts[transformation] = transformation_counts.get(transformation, 0) + 1
        
        return {
            'total_transformations': len(self.transformations_applied),
            'unique_transformations': len(set(self.transformations_applied)),
            'transformation_counts': transformation_counts,
            'available_transformations': self.jdt_transformer.get_available_transformations('enhanced'),
            'disabled_transformations': self.disabled_transformations
        }

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Enhanced semantic augmentation using JDT')
    parser.add_argument('input', help='Input Java file or directory')
    parser.add_argument('output', help='Output file or directory')
    parser.add_argument('--variants', type=int, default=3, help='Number of variants to create')
    parser.add_argument('--sequence-len', type=int, default=2, help='Transformations per variant sequence')
    parser.add_argument('--max-depth', type=int, default=3, help='Max transformation depth per file')
    parser.add_argument('--avoid', nargs='*', default=[], help='Transformations to avoid')
    parser.add_argument('--min-diff', type=float, default=0.03, help='Minimal token diff ratio to accept a variant')
    parser.add_argument('--focus-nodes', nargs='*', default=['control','dataflow'], help='Bias toward nodes impacting these aspects')
    parser.add_argument('--manifest', action='store_true', default=True, help='Write per-variant manifests and index')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--disabled', nargs='*', default=[], help='Disabled transformations')
    parser.add_argument('--jdt-jar', help='Path to JDT transformer JAR')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--require-change', action='store_true', help='Only emit variants whose body changes vs original')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize transformer
        transformer = EnhancedSemanticTransformer(
            seed=args.seed,
            disabled_transformations=args.disabled,
            jdt_jar_path=args.jdt_jar
        )
        
        # Process input
        if os.path.isfile(args.input):
            # Single file
            transformed_code = transformer.transform_file(args.input, 0)
            if args.require_change:
                try:
                    with open(args.input, 'r') as f:
                        original_code = f.read()
                    # Remove header from transformed only; original has no header
                    tl = transformed_code.splitlines()
                    transformed_body = "\n".join(tl[4:] if len(tl) >= 4 and tl[0].startswith('/*') else tl)
                    if transformed_body == original_code:
                        logger.info("No body change detected; --require-change set, skipping write")
                        return 0
                except Exception:
                    pass
            
            with open(args.output, 'w') as f:
                f.write(transformed_code)
            
            print(f"Transformed file: {args.input} -> {args.output}")
            
        elif os.path.isdir(args.input):
            # Directory
            if not args.require_change:
                results = transformer.transform_directory(
                    args.input, args.output, args.variants,
                    sequence_len=args.sequence_len, max_depth=args.max_depth,
                    avoid=args.avoid, min_diff=args.min_diff,
                    focus_nodes=args.focus_nodes, write_manifest=args.manifest)
            else:
                # Require-change mode: write only when body changes
                in_path = Path(args.input)
                out_path = Path(args.output)
                out_path.mkdir(parents=True, exist_ok=True)
                files_processed = 0
                variants_created = 0
                errors = []
                java_files = list(in_path.glob('**/*.java'))
                logger.info(f"Found {len(java_files)} Java files to process (require-change mode)")
                for jf in java_files:
                    try:
                        rel = jf.relative_to(in_path)
                        with open(jf, 'r') as f:
                            original = f.read()
                        for vid in range(args.variants):
                            transformed = transformer.transform_file(
                                str(jf), vid,
                                sequence_len=args.sequence_len, max_depth=args.max_depth,
                                avoid=args.avoid, min_diff=args.min_diff, focus_nodes=args.focus_nodes)
                            tl = transformed.splitlines()
                            transformed_body = tl[4:] if len(tl) >= 4 and tl[0].startswith('/*') else tl
                            # Compare transformed body against full original
                            if transformed_body == original.splitlines():
                                # If identical (ignoring header size heuristic), skip
                                continue
                            out_file = out_path / rel.parent / f"{jf.stem}_variant_{vid}{jf.suffix}"
                            out_file.parent.mkdir(parents=True, exist_ok=True)
                            with open(out_file, 'w') as f:
                                f.write(transformed)
                            variants_created += 1
                        files_processed += 1
                    except Exception as e:
                        errors.append(f"{jf}: {e}")
                results = {
                    'files_processed': files_processed,
                    'variants_created': variants_created,
                    'errors': errors
                }
            
            print(f"Processing complete:")
            print(f"  Files processed: {results['files_processed']}")
            print(f"  Variants created: {results['variants_created']}")
            print(f"  Errors: {len(results['errors'])}")
            
            if results['errors']:
                print("Errors encountered:")
                for error in results['errors']:
                    print(f"  - {error}")
        
        else:
            raise FileNotFoundError(f"Input path not found: {args.input}")
        
        # Print transformation statistics
        stats = transformer.get_transformation_stats()
        print(f"\nTransformation Statistics:")
        print(f"  Total transformations applied: {stats['total_transformations']}")
        print(f"  Unique transformation types: {stats['unique_transformations']}")
        print(f"  Available transformations: {len(stats['available_transformations'])}")
        
        if stats['transformation_counts']:
            print("  Most used transformations:")
            for trans, count in sorted(stats['transformation_counts'].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {trans}: {count}")
        
    except Exception as e:
        logger.error(f"Transformation failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())