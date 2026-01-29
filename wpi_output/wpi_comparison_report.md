# WPI vs Model-Based Annotation Placement Comparison

**Generated**: 2026-01-22
**Checker Framework Version**: 3.42.0

## Summary

This report compares Whole Program Inference (WPI) results with model-based annotation placement results for the Lower Bound Checker evaluation.

## Important Methodology Notes

### WPI Configuration
- WPI was run using the `IndexChecker` which is a composite checker including:
  - LowerBoundChecker
  - UpperBoundChecker
  - SameLenChecker  
  - LessThanChecker
  - ValueChecker
  - SubstringIndexChecker
  - SearchIndexChecker
- WPI iteratively infers annotations until convergence
- Output: `.ajava` files containing inferred annotations

### Model-Based Configuration
- Models were trained for specific annotation types
- Evaluation was run with Maven dependency resolution
- Output: Direct annotations inserted into source code

## WPI Results

| Project | Iterations | Ajava Files | Total Warnings | LowerBound Warnings |
|---------|------------|-------------|----------------|---------------------|
| sortpom | 8 | 378 | 33 | 5 |
| pom-tuner | 8 | 147 | 91 | 15 |
| eclipse-m2e | SKIPPED | - | - | - |

**Note**: eclipse-external-annotations-m2e-plugin uses Tycho (Eclipse/OSGi build system) which has a different compilation process incompatible with the standard WPI procedure.

## Model-Based Results (from evaluation_report.json)

| Project | Baseline Warnings | Best Model | After Warnings | Reduction % |
|---------|-------------------|------------|----------------|-------------|
| sortpom | 2 | gcn/hgt/causal/etc | 1 | 50.0% |
| pom-tuner | 38 | gcn/hgt/causal/etc | 6 | 84.2% |
| eclipse-m2e | 83 | gcn/hgt/causal/etc | 88 | -6.0% |

## Comparison Analysis

### sortpom
- **WPI**: 5 LowerBound warnings remaining, 378 ajava files generated
- **Models**: 1 warning after placement (from baseline of 2)
- **Analysis**: WPI with full IndexChecker found more potential issues than the focused LowerBound model approach. The models achieved 50% reduction on the specific LowerBound warnings they targeted.

### pom-tuner  
- **WPI**: 15 LowerBound warnings remaining, 147 ajava files generated
- **Models**: 6 warnings after placement (from baseline of 38)
- **Analysis**: Models achieved 84% reduction. WPI's remaining 15 warnings are after iterative inference convergence, suggesting these require manual annotation or code changes.

### eclipse-external-annotations-m2e-plugin
- **WPI**: Not applicable (Tycho build system)
- **Models**: -6% (warnings increased from 83 to 88)
- **Analysis**: Model predictions on this project were incorrect, possibly due to training data mismatch or complexity of Eclipse plugin development patterns.

## Key Differences

| Aspect | WPI | Model-Based |
|--------|-----|-------------|
| Approach | Iterative inference | Single-pass prediction |
| Output format | .ajava files | Direct source modifications |
| Checker scope | Full IndexChecker | Specific checker focus |
| Build system support | Maven (standard) | Maven with dependency resolution |
| Convergence | Iterates until stable | One-time placement |

## Conclusions

1. **WPI produces comprehensive annotations** across multiple index-related checkers, while models focus on specific annotation types.

2. **Both approaches have limitations**: WPI cannot reduce all warnings (some require code changes), and models can make incorrect predictions that increase warnings.

3. **Tycho/Eclipse projects require special handling** for both approaches.

4. **For fair comparison**, the same checker configuration should be used for both baseline and post-annotation evaluation.

## Recommendations

1. For projects requiring comprehensive index safety, WPI with IndexChecker is more thorough
2. For targeted warning reduction (e.g., LowerBound only), model-based approaches can be effective
3. Consider using WPI output to augment model training data
4. Handle Eclipse/Tycho projects with specialized tooling

## Files Generated

- `sortpom-wpi/` - 378 ajava files with inferred annotations
- `pom-tuner-wpi/` - 147 ajava files with inferred annotations
- `sortpom_typecheck.out` - Full typecheck output for sortpom
- `pom-tuner_typecheck.out` - Full typecheck output for pom-tuner
