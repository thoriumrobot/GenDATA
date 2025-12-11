# GitHub Project Discovery Guide

This guide explains how to use the project discovery pipeline to find GitHub Java projects that are likely to compile and produce Lower Bound Checker warnings.

## Overview

The project discovery pipeline consists of five phases:

1. **GitHub Search**: Search GitHub for Java projects using various criteria
2. **Pattern Analysis**: Analyze code patterns that trigger Lower Bound Checker warnings
3. **Compilation Test**: Test whether projects compile successfully
4. **Warning Test**: Run Lower Bound Checker and count warnings
5. **Scoring**: Score and rank projects based on multiple criteria

## Prerequisites

### Required Tools

- Python 3.7+
- Git
- Java JDK 8+
- Maven or Gradle (for compilation testing)
- Checker Framework (for warning testing)

### Python Dependencies

```bash
pip install PyGithub
```

### Environment Variables

Optional but recommended:

- `GITHUB_TOKEN`: GitHub personal access token (increases API rate limit from 60 to 5000 requests/hour)
- `CHECKERFRAMEWORK_CP`: Checker Framework classpath

## Quick Start

### Run Complete Pipeline

```bash
python3 find_lower_bound_projects.py \
    --max-projects 50 \
    --min-stars 10 \
    --min-score 70 \
    --github-token YOUR_GITHUB_TOKEN
```

### Run Individual Phases

#### Phase 1: GitHub Search

```bash
python3 github_project_finder.py \
    --use-all-queries \
    --max-results 100 \
    --min-stars 10 \
    --output github_projects.json \
    --github-token YOUR_GITHUB_TOKEN
```

#### Phase 2: Pattern Analysis

```bash
python3 analyze_code_patterns.py \
    --input github_projects.json \
    --output pattern_analysis.json \
    --max-projects 50
```

#### Phase 3: Compilation Test

```bash
python3 test_project_compilation.py \
    --input pattern_analysis.json \
    --output compilation_results.json \
    --timeout 600
```

#### Phase 4: Warning Test

```bash
python3 test_lower_bound_warnings.py \
    --input compilation_results.json \
    --output warning_test_results.json \
    --max-files 100 \
    --checker-cp $CHECKERFRAMEWORK_CP
```

#### Phase 5: Scoring

```bash
python3 score_projects.py \
    --github-projects github_projects.json \
    --pattern-analysis pattern_analysis.json \
    --compilation-results compilation_results.json \
    --warning-results warning_test_results.json \
    --output ranked_projects.json \
    --min-score 70
```

## Detailed Usage

### GitHub Project Finder

Searches GitHub for Java projects matching criteria.

**Options:**
- `--query`: Custom GitHub search query
- `--use-all-queries`: Use all built-in search queries
- `--max-results`: Maximum results per query
- `--min-stars`: Minimum GitHub stars
- `--min-size`: Minimum repository size (KB)
- `--max-size`: Maximum repository size (KB)
- `--updated-within-days`: Only projects updated within N days
- `--github-token`: GitHub API token

**Example:**
```bash
python3 github_project_finder.py \
    --query "language:java stars:>20 pom.xml" \
    --max-results 50 \
    --output projects.json
```

### Code Pattern Analyzer

Analyzes Java code for patterns that trigger Lower Bound Checker warnings.

**Patterns Detected:**
- Array access operations (`array[index]`)
- Loop variables (`for (int i = ...)`)
- Array length operations (`.length`)
- Comparisons with 0 and -1
- Array creation (`new Type[size]`)
- Index variable usage
- Array bounds checks
- Negative index access

**Options:**
- `--input`: Input JSON file with projects
- `--output`: Output JSON file
- `--max-projects`: Maximum projects to analyze
- `--temp-dir`: Temporary directory for cloning

**Example:**
```bash
python3 analyze_code_patterns.py \
    --input github_projects.json \
    --output patterns.json \
    --max-projects 20
```

### Compilation Tester

Tests whether projects compile successfully.

**Supported Build Systems:**
- Maven (`pom.xml`)
- Gradle (`build.gradle`, `gradlew`)
- Ant (`build.xml`)
- Make (`Makefile`)

**Options:**
- `--input`: Input JSON file with projects
- `--output`: Output JSON file
- `--max-projects`: Maximum projects to test
- `--timeout`: Compilation timeout (seconds)
- `--temp-dir`: Temporary directory for cloning

**Example:**
```bash
python3 test_project_compilation.py \
    --input pattern_analysis.json \
    --output compiled.json \
    --timeout 300
```

### Warning Tester

Runs Lower Bound Checker on projects and counts warnings.

**Options:**
- `--input`: Input JSON file with projects
- `--output`: Output JSON file
- `--max-projects`: Maximum projects to test
- `--max-files`: Maximum Java files per project
- `--checker-cp`: Checker Framework classpath
- `--timeout`: Timeout (seconds)
- `--temp-dir`: Temporary directory for cloning

**Example:**
```bash
python3 test_lower_bound_warnings.py \
    --input compilation_results.json \
    --output warnings.json \
    --max-files 50 \
    --checker-cp $CHECKERFRAMEWORK_CP
```

### Project Scorer

Scores and ranks projects based on multiple criteria.

**Scoring Criteria:**
- **Compilation Success** (30 points): Must compile successfully
- **Warning Count** (30 points): Ideal range 50-500 warnings
- **Pattern Density** (20 points): High pattern density indicates likely warnings
- **Project Size** (10 points): Ideal 100-1000 Java files

**Options:**
- `--github-projects`: GitHub projects JSON file
- `--pattern-analysis`: Pattern analysis JSON file
- `--compilation-results`: Compilation results JSON file
- `--warning-results`: Warning test results JSON file
- `--combined-input`: Single combined input file (alternative)
- `--output`: Output JSON file
- `--min-warnings`: Minimum warnings required
- `--max-warnings`: Maximum warnings allowed
- `--min-score`: Minimum score to include

**Example:**
```bash
python3 score_projects.py \
    --github-projects github_projects.json \
    --pattern-analysis pattern_analysis.json \
    --compilation-results compilation_results.json \
    --warning-results warning_test_results.json \
    --output ranked.json \
    --min-score 70
```

## Output Format

### Final Output (`ranked_projects.json`)

```json
{
  "metadata": {
    "total_projects": 25,
    "high_score_projects": 10,
    "generated_at": "2024-01-01T12:00:00"
  },
  "ranked_projects": [
    {
      "project_name": "owner/repo",
      "project_url": "https://github.com/owner/repo",
      "total_score": 85.0,
      "compilation_score": 30.0,
      "warning_score": 30.0,
      "pattern_score": 20.0,
      "size_score": 10.0,
      "compilation_success": true,
      "warning_count": 342,
      "pattern_density": 0.12,
      "java_files": 250,
      "build_system": "maven"
    }
  ]
}
```

## Filtering Criteria

### Must Have

- ✅ Java project (primary language)
- ✅ Build file (pom.xml, build.gradle, etc.)
- ✅ Compiles successfully
- ✅ 10+ Lower Bound warnings
- ✅ Open source license

### Should Have

- ⭐ 10+ stars (indicates quality)
- ⭐ Updated within 2 years (more likely to compile)
- ⭐ 100-1000 Java files (manageable size)
- ⭐ Moderate pattern density (0.05-0.2 patterns/LOC)
- ⭐ Clean code structure

## Troubleshooting

### GitHub API Rate Limits

If you hit rate limits:
- Use `--github-token` with a personal access token
- Reduce `--max-results` per query
- Run phases separately with delays

### Compilation Failures

Common issues:
- Missing dependencies
- Wrong Java version
- Build system not detected
- Timeout too short

Solutions:
- Increase `--timeout`
- Check build system manually
- Skip problematic projects

### Checker Framework Issues

If warnings aren't generated:
- Verify `CHECKERFRAMEWORK_CP` is set correctly
- Check Checker Framework installation
- Ensure Java files are compilable
- Check `--max-files` limit

## Best Practices

1. **Start Small**: Test with `--max-projects 10` first
2. **Use GitHub Token**: Significantly increases rate limits
3. **Run Phases Separately**: Easier to debug and resume
4. **Check Intermediate Files**: Verify each phase's output
5. **Monitor Disk Space**: Cloning repositories uses space
6. **Clean Up**: Remove temporary directories after completion

## Example Workflow

```bash
# 1. Set up environment
export GITHUB_TOKEN="your_token_here"
export CHECKERFRAMEWORK_CP="/path/to/checker/framework/classpath"

# 2. Run complete pipeline
python3 find_lower_bound_projects.py \
    --max-projects 50 \
    --min-stars 15 \
    --min-score 75 \
    --work-dir ./discovery_work

# 3. Review results
cat discovery_work/lower_bound_project_candidates.json | jq '.ranked_projects[] | {name: .project_name, score: .total_score, warnings: .warning_count}'

# 4. Manually review top projects
# Check project URLs, verify compilation, review warnings
```

## Integration with GenDATA

Discovered projects can be added to `case_studies/` directory:

1. Clone the repository:
   ```bash
   cd case_studies
   git clone https://github.com/owner/repo.git
   ```

2. Generate warnings:
   ```bash
   python3 checker_framework_runner.py \
       case_studies/repo \
       case_studies/repo_lower_bound_warnings.out \
       --checker-name lower_bound
   ```

3. Run predictions:
   ```bash
   python3 simple_annotation_type_pipeline.py \
       --warnings-file case_studies/repo_lower_bound_warnings.out \
       --project-root case_studies/repo
   ```

## Limitations

- GitHub API rate limits (mitigated with token)
- Compilation may fail due to missing dependencies
- Some projects may have too many/few warnings
- Pattern analysis is approximate (regex-based)
- Large projects may timeout

## Future Improvements

- AST-based pattern analysis (more accurate)
- Parallel processing for faster analysis
- Caching of cloned repositories
- Support for more build systems
- Better error handling and recovery
- Integration with project metadata APIs

