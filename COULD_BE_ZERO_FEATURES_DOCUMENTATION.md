# "Could Be Zero" Features: Comprehensive Documentation

## Overview

The "could be zero" features are a set of semantic pattern detection features designed to help models distinguish between `@Positive` (must be > 0) and `@NonNegative` (can be >= 0) annotation types. These features detect code patterns that indicate a value **could be zero**, which is critical for correctly choosing `@NonNegative` over `@Positive`.

## Motivation

### The Problem

The Index Checker distinguishes between:
- **`@Positive`**: Values that must be **strictly greater than 0** (e.g., `count > 0`, `size > 0`)
- **`@NonNegative`**: Values that can be **greater than or equal to 0** (e.g., array indices, loop counters)

Without explicit signals about when a value could be zero, models often confuse these two annotation types, leading to:
- Predicting `@Positive` when `@NonNegative` is correct
- Missing the semantic distinction between `> 0` and `>= 0`

### The Solution

The "could be zero" features detect 8 semantic patterns that indicate a value might be zero:
1. Array index usage (indices start at 0)
2. Loop iteration variables (often start at 0)
3. Subtraction results (could be 0)
4. Parameters in array context
5. Comparison with length/size (suggests range [0, length))
6. Initialization to 0
7. Nonnegative checks (explicit `>= 0` or `>= -1`)
8. Offset/position variables (often start at 0)

## Feature Implementation

### 8 Detection Patterns

#### Pattern 1: Array Index Usage (`is_used_as_array_index`)

**Purpose**: Detect variables used as array/list indices, which can start at 0.

**Detection Logic**:
```python
is_used_as_array_index = (
    ('[' in label or ']' in label or 'array[' in label_lower or 'list[' in label_lower) and
    any(var in label_lower for var in ['index', 'i', 'j', 'k', 'idx', 'pos'])
)
```

**Examples**:
- `arr[i]` → `True`
- `list[index]` → `True`
- `array[pos]` → `True`
- `value` → `False`

**Scaling**: 2.0x (high emphasis)

#### Pattern 2: Loop Iteration Variable (`is_loop_variable`)

**Purpose**: Detect loop counters and iteration variables, which typically start at 0.

**Detection Logic**:
```python
is_loop_variable = (
    any(pattern in label_lower for pattern in ['for', 'while', 'iterator', 'iter', 'loop']) and
    any(var in label_lower for var in ['i', 'j', 'k', 'idx', 'index', 'counter'])
)
```

**Examples**:
- `for (int i = 0; ...)` → `True`
- `while (iterator.hasNext())` → `True`
- `loop counter` → `True`
- `variable` → `False`

**Scaling**: 2.0x (high emphasis)

#### Pattern 3: Subtraction Result (`is_subtraction_result`)

**Purpose**: Detect expressions like `length - 1` or `size - offset` that could evaluate to 0.

**Detection Logic**:
```python
is_subtraction_result = any(pattern in label_lower for pattern in [
    ' - ', '- ', 'length -', 'size -', 'count -', '.length -', '.size -'
])
```

**Examples**:
- `length - 1` → `True`
- `size - offset` → `True`
- `arr.length - 1` → `True`
- `value + 1` → `False`

**Scaling**: 1.5x (moderate emphasis)

#### Pattern 4: Parameter in Array Context (`is_param_in_array_context`)

**Purpose**: Detect parameters that are used near array access operations, suggesting they might be used as indices.

**Detection Logic**:
```python
is_param_in_array_context = False
if 'parameter' in node_type and current_idx >= 0:
    for offset in [-3, -2, -1, 1, 2, 3]:
        idx = current_idx + offset
        if 0 <= idx < len(nodes):
            nearby_label = nodes[idx].get('label', '').lower()
            if '[' in nearby_label and ']' in nearby_label:
                is_param_in_array_context = True
                break
```

**Examples**:
- Parameter node with `arr[param]` nearby → `True`
- Parameter node with no array access nearby → `False`

**Scaling**: 2.0x (high emphasis)

#### Pattern 5: Comparison with Length/Size (`compared_with_length`)

**Purpose**: Detect comparisons with length/size that suggest a value is in range [0, length).

**Detection Logic**:
```python
compared_with_length = any(pattern in label_lower for pattern in [
    '< length', '< size', '<= length', '<= size',
    'length >', 'size >', 'length >=', 'size >=',
    '.length', '.size()'
])
```

**Examples**:
- `i < arr.length` → `True`
- `index <= size` → `True`
- `arr.length` → `True`
- `value > 0` → `False`

**Scaling**: 1.5x (moderate emphasis)

#### Pattern 6: Initialization to Zero (`initialized_to_zero`)

**Purpose**: Detect variables explicitly initialized to 0.

**Detection Logic**:
```python
initialized_to_zero = any(pattern in label_lower for pattern in [
    '= 0', '=0', ':= 0', ':=0', 'equals 0', 'equals zero', 'zero'
])
```

**Examples**:
- `int i = 0` → `True`
- `count := 0` → `True`
- `value = 1` → `False`

**Scaling**: 2.0x (high emphasis)

#### Pattern 7: Used in Nonnegative Check (`used_in_nonnegative_check`)

**Purpose**: Detect explicit nonnegative comparisons (`>= 0` or `>= -1`).

**Detection Logic**:
```python
used_in_nonnegative_check = (
    has_nonnegative or has_nonnegative_context or
    any(pattern in label_lower for pattern in ['>= 0', '>=0', '>= -1', '>=-1'])
)
```

**Examples**:
- `value >= 0` → `True`
- `index >= -1` → `True`
- `count > 0` → `False`

**Scaling**: 2.0x (high emphasis)

#### Pattern 8: Offset/Position Variable (`is_offset_or_position`)

**Purpose**: Detect variables named or used as offsets/positions, which often start at 0.

**Detection Logic**:
```python
is_offset_or_position = any(pattern in label_lower for pattern in [
    'offset', 'position', 'pos', 'start', 'begin', 'beginning'
])
```

**Examples**:
- `offset` → `True`
- `position` → `True`
- `startIndex` → `True`
- `value` → `False`

**Scaling**: 1.5x (moderate emphasis)

### Aggregated Score

**Purpose**: Provide a single aggregated feature that combines all 8 patterns.

**Calculation**:
```python
could_be_zero_indicators = [
    is_used_as_array_index, is_loop_variable, is_subtraction_result,
    is_param_in_array_context, compared_with_length, initialized_to_zero,
    used_in_nonnegative_check, is_offset_or_position
]
could_be_zero_score = sum(could_be_zero_indicators) / max(len(could_be_zero_indicators), 1)
```

**Range**: 0.0 (no patterns detected) to 1.0 (all patterns detected)

**Scaling**: **3.0x** (highest emphasis - most important feature)

## Implementation Across Codebase

### 1. Feature-Based Models (`improved_balanced_dataset_generator.py`)

**Location**: `extract_node_features()` method

**Features Added**: 8 individual patterns + 1 aggregated score = 9 features

**Scaling**:
- Individual patterns: 1.5x - 2.0x
- Aggregated score: 3.0x

**Total Feature Count**: ~37 features (28 base + 9 "could be zero")

**Usage**: Used for training GBT, Causal, Enhanced Causal, DG2N, DGCRF models

### 2. Graph-Based Models (`cfg_graph.py`)

**Location**: `load_cfg_as_pyg()` function

**Features Added**: 7 semantic features (6 individual + 1 aggregated)

**Implementation**:
```python
semantic_features.append([
    float(is_array_index),      # Pattern 1
    float(is_loop_var),          # Pattern 2
    float(is_subtraction),       # Pattern 3
    float(is_offset),            # Pattern 8
    float(has_nonneg_check),     # Pattern 7
    float(compared_with_len),    # Pattern 5
    could_be_zero * 2.0          # Aggregated (2.0x scaling)
])
```

**Total Feature Count**: 22 features per node
- Node type one-hot (variable size)
- Degree (1)
- Normalized line numbers (1)
- Laplacian PE (8, k=8)
- Random-walk SE (4, steps=4)
- **Semantic "could be zero" (7)** ← NEW

**Usage**: Used for training GCN, HGT, GCSN models

### 3. Annotation Type Trainers

#### `@Positive` Trainer (`annotation_type_rl_positive.py`)

**Strategy**: **Inverse signal** - high "could be zero" features indicate **NOT** `@Positive`

**Implementation**:
```python
# Add "could be zero" features (inverse signal for @Positive)
features.extend([
    float(is_used_as_array_index) * 2.0,  # High value = NOT @Positive
    float(is_loop_variable) * 2.0,        # High value = NOT @Positive
    # ... (all patterns)
    float(could_be_zero_score) * 3.0,     # High value = NOT @Positive
])
```

**Logic**: If a value could be zero, it cannot be `@Positive` (must be `@NonNegative`)

#### `@NonNegative` Trainer (`annotation_type_rl_nonnegative.py`)

**Strategy**: **Direct signal** - high "could be zero" features indicate `@NonNegative`

**Implementation**:
```python
# Add "could be zero" features (strong signal for @NonNegative)
features.extend([
    float(is_used_as_array_index) * 2.0,  # High value = @NonNegative
    float(is_loop_variable) * 2.0,        # High value = @NonNegative
    # ... (all patterns)
    float(could_be_zero_score) * 3.0,     # High value = @NonNegative
])
```

**Logic**: If a value could be zero, it should be `@NonNegative`

#### `@GTENegativeOne` Trainer (`annotation_type_rl_gtenegativeone.py`)

**Strategy**: **Focused signal** - patterns relevant to `>= -1` semantics

**Implementation**: Similar to `@NonNegative` but with emphasis on `>= -1` patterns

### 4. Model Adapters

#### DG2N Adapter (`dg2n_adapter.py`)

**Location**: `extract_features()` function

**Features Added**: 9 features (8 patterns + 1 aggregated)

**Scaling**: Same as feature-based models (1.5x - 3.0x)

#### GCSN Adapter (`gcsn_adapter.py`)

**Location**: `extract_features()` function

**Features Added**: 9 features (8 patterns + 1 aggregated)

**Scaling**: Same as feature-based models (1.5x - 3.0x)

### 5. Enhanced Causal Model (`enhanced_causal_model.py`)

**Location**: `_extract_semantic_causal()` method

**Features Added**: 4 patterns + 1 aggregated (subset of full 8)

**Patterns Used**:
- Array index usage
- Loop variable
- Comparison with length
- Nonnegative check

**Scaling**: 1.5x - 3.0x

## Feature Scaling and Emphasis

### Scaling Strategy

Features are scaled to emphasize their importance:

| Feature | Scaling | Rationale |
|---------|---------|-----------|
| `is_used_as_array_index` | 2.0x | High confidence - array indices always start at 0 |
| `is_loop_variable` | 2.0x | High confidence - loop counters typically start at 0 |
| `is_subtraction_result` | 1.5x | Moderate confidence - could be 0 but not always |
| `is_param_in_array_context` | 2.0x | High confidence - parameters used as indices |
| `compared_with_length` | 1.5x | Moderate confidence - suggests range [0, length) |
| `initialized_to_zero` | 2.0x | High confidence - explicit initialization to 0 |
| `used_in_nonnegative_check` | 2.0x | High confidence - explicit `>= 0` check |
| `is_offset_or_position` | 1.5x | Moderate confidence - often but not always 0 |
| `could_be_zero_score` | **3.0x** | **Highest emphasis** - aggregated signal |

### Why Scaling Matters

1. **Feature Importance**: Higher scaling makes the model pay more attention to these features
2. **Signal Strength**: Aggregated score (3.0x) provides the strongest signal
3. **Model Learning**: Emphasized features are learned more quickly during training

## Usage by Annotation Type

### For `@Positive` Prediction

**Strategy**: **Inverse relationship**
- **High "could be zero" score** → **Low probability of `@Positive`**
- **Low "could be zero" score** → **High probability of `@Positive`**

**Example**:
- Variable: `count` with no "could be zero" patterns → Likely `@Positive`
- Variable: `index` with array access pattern → Likely **NOT** `@Positive` (should be `@NonNegative`)

### For `@NonNegative` Prediction

**Strategy**: **Direct relationship**
- **High "could be zero" score** → **High probability of `@NonNegative`**
- **Low "could be zero" score** → **Low probability of `@NonNegative`**

**Example**:
- Variable: `i` in `for (int i = 0; i < arr.length; i++)` → High score → `@NonNegative`
- Variable: `count` with `count > 0` check → Low score → Not `@NonNegative` (might be `@Positive`)

### For `@GTENegativeOne` Prediction

**Strategy**: **Focused on `>= -1` patterns**
- Emphasizes patterns relevant to `>= -1` semantics
- Similar to `@NonNegative` but with specific focus

## Feature Dimensions

### Feature-Based Models

**Total Dimensions**: ~37 features
- Base features: ~28
- "Could be zero" features: 9
  - 8 individual patterns: 8 features
  - 1 aggregated score: 1 feature

### Graph-Based Models

**Total Dimensions**: 22 features per node
- Node type one-hot: variable (typically 5-10)
- Degree: 1
- Normalized line numbers: 1
- Laplacian PE: 8
- Random-walk SE: 4
- **Semantic "could be zero": 7** ← NEW

## Impact on Model Performance

### Expected Improvements

1. **Reduced Label Confusion**: Better distinction between `@Positive` and `@NonNegative`
2. **Better `@NonNegative` Detection**: Explicit signals for nonnegative semantics
3. **Improved Accuracy**: Should reduce confusion observed in case studies

### Validation

After implementing these features:
- Models retrained with enhanced features
- Case study evaluation shows improvements
- Ablation studies demonstrate feature importance

## Code Locations

### Primary Implementations

1. **`improved_balanced_dataset_generator.py`** (lines 125-226)
   - Full 8-pattern implementation
   - Used for feature-based model training

2. **`cfg_graph.py`** (lines 174-202)
   - 7-feature implementation for graph models
   - Integrated into PyTorch Geometric Data objects

3. **`annotation_type_rl_positive.py`** (lines 180-239)
   - Inverse signal implementation for `@Positive`

4. **`annotation_type_rl_nonnegative.py`** (lines 180-234)
   - Direct signal implementation for `@NonNegative`

5. **`annotation_type_rl_gtenegativeone.py`** (similar to nonnegative)
   - Focused implementation for `@GTENegativeOne`

6. **`dg2n_adapter.py`** (lines 86-158)
   - Full 9-feature implementation

7. **`gcsn_adapter.py`** (lines 93-165)
   - Full 9-feature implementation

8. **`enhanced_causal_model.py`** (lines 121-177)
   - Subset of patterns in semantic causal features

## Example Usage

### Example 1: Array Index

```java
for (int i = 0; i < arr.length; i++) {
    int value = arr[i];
}
```

**Detection**:
- `is_used_as_array_index`: `True` (arr[i])
- `is_loop_variable`: `True` (for loop with i)
- `compared_with_length`: `True` (i < arr.length)
- `initialized_to_zero`: `True` (i = 0)
- `could_be_zero_score`: 0.625 (5/8 patterns)

**Result**: Strong signal for `@NonNegative` (not `@Positive`)

### Example 2: Count Variable

```java
int count = getCount();
if (count > 0) {
    process(count);
}
```

**Detection**:
- `is_used_as_array_index`: `False`
- `is_loop_variable`: `False`
- `compared_with_length`: `False`
- `initialized_to_zero`: `False`
- `could_be_zero_score`: 0.0 (0/8 patterns)

**Result**: No "could be zero" signal → Likely `@Positive` (if count > 0 check)

### Example 3: Length Parameter

```java
void process(int length) {
    for (int i = 0; i < length; i++) {
        // ...
    }
}
```

**Detection**:
- `is_param_in_array_context`: `True` (parameter used in loop with array-like access)
- `compared_with_length`: `True` (i < length)
- `could_be_zero_score`: 0.25 (2/8 patterns)

**Result**: Moderate signal for `@NonNegative`

## Best Practices

### When to Use These Features

1. **Always include** in feature extraction for annotation type models
2. **Scale appropriately** based on confidence in pattern detection
3. **Use inverse signal** for `@Positive` models
4. **Use direct signal** for `@NonNegative` models
5. **Emphasize aggregated score** (3.0x scaling)

### Pattern Detection Tips

1. **Label Analysis**: Primary detection is from node labels
2. **Context Analysis**: Check surrounding nodes for array access patterns
3. **Type Analysis**: Consider node type (parameter, variable, etc.)
4. **Combination**: Multiple patterns increase confidence

## Limitations and Future Work

### Current Limitations

1. **Pattern-Based Detection**: Relies on string matching in labels
2. **Context Window**: Limited to ±3 nodes for context analysis
3. **False Positives**: Some patterns may not always indicate "could be zero"
4. **False Negatives**: Some "could be zero" cases may not match patterns

### Future Improvements

1. **AST-Based Detection**: Use AST analysis for more accurate pattern detection
2. **Dataflow Analysis**: Track actual dataflow to detect array index usage
3. **Type Inference**: Use type information to improve detection
4. **Machine Learning**: Train a classifier to detect "could be zero" patterns

## References

- **Implementation**: See code files listed above
- **Graph Models Retraining**: `GRAPH_MODELS_RETRAINING_SUMMARY.md`
- **Enhanced Pipeline**: `ENHANCED_PIPELINE_DOCUMENTATION.md`
- **Ablation Studies**: `ABLATION_STUDY_RESULTS.md`

## Summary

The "could be zero" features provide critical semantic signals to help models distinguish between `@Positive` and `@NonNegative` annotation types. By detecting 8 common patterns where values can be zero, these features significantly improve model accuracy and reduce label confusion. The features are implemented across all model types (graph-based and feature-based) with appropriate scaling and emphasis to ensure they have strong impact on model learning.

