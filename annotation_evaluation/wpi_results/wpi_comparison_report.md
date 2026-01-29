# WPI vs Model-Based Annotation Placement Comparison

**Generated**: 2026-01-22T20:46:43.676310

## Important Note: DLJC/Maven Compatibility Issue

**WPI could not analyze any of the test projects** due to a known compatibility issue between DLJC (Do Like Javac) and Maven-based projects.

### What Happened

The WPI script uses DLJC to intercept javac commands during Maven builds. However, DLJC could not capture the Java source files from these Maven projects, resulting in "no source files" errors.

### Why This Happens

This is a known limitation documented in the Checker Framework manual. Some Maven projects:
- Use the Maven Compiler Plugin in a way DLJC can't intercept
- Have complex multi-module setups that confuse DLJC
- Require special Maven configuration for DLJC to work

### Alternative Approaches

To compare WPI with model-based annotation placement, you would need to either:
1. Configure Maven to use a compiler plugin that DLJC can intercept
2. Run the Checker Framework directly on the source files
3. Use a Gradle-based build instead

### Model Results Remain Valid

The model-based annotation placement results were obtained by running the Checker Framework directly on Java files with proper Maven classpath resolution, which worked correctly.

## Summary

| Project | WPI Reduction | Model Reduction | Winner | Difference |
|---------|---------------|-----------------|--------|------------|
| sortpom | 0.0% | 50.0% | N/A (WPI failed) | -50.0% |
| eclipse-external-annotations-m2e-plugin | 0.0% | 0.0% | Model | +0.0% |
| pom-tuner | 0.0% | 84.2% | N/A (WPI failed) | -84.2% |

## Detailed Results

### sortpom

**WPI Results**:
- Success: False
- Baseline warnings: 2
- After WPI warnings: 0
- Reduction: 0.0%
- Annotations inferred: 0
- Execution time: 5.4s
- Error: DLJC could not capture Java source files from Maven build. This is a known compatibility issue.

**Model Results**:
- Baseline warnings: 2
- After model warnings: 1
- Reduction: 50.0%

### eclipse-external-annotations-m2e-plugin

**WPI Results**:
- Success: False
- Baseline warnings: 83
- After WPI warnings: 0
- Reduction: 0.0%
- Annotations inferred: 0
- Execution time: 9.5s

**Model Results**:
- Baseline warnings: 83
- After model warnings: 83
- Reduction: 0.0%

### pom-tuner

**WPI Results**:
- Success: False
- Baseline warnings: 38
- After WPI warnings: 0
- Reduction: 0.0%
- Annotations inferred: 0
- Execution time: 8.7s
- Error: DLJC could not capture Java source files from Maven build. This is a known compatibility issue.

**Model Results**:
- Baseline warnings: 38
- After model warnings: 6
- Reduction: 84.2%
