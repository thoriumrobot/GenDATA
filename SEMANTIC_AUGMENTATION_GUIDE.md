# Semantic Augmentation Guide

## Overview

The semantic augmentation system replaces the previous random code injection approach with semantic-preserving transformations that slicers are less likely to remove. This approach maintains the original semantics while changing the syntactic structure, making the augmented data more valuable for training machine learning models.

## Key Transformations

### 1. Loop Conversions
**For ↔ While Loop Conversion**
```java
// Original for loop
for (int i = 0; i < array.length; i++) {
    sum += array[i];
}

// Transformed to while loop
int i = 0;
while (i < array.length) {
    sum += array[i];
    i++;
}
```

### 2. Guard Reversals
**If-Else Condition Flipping**
```java
// Original
if (sum > 0) {
    System.out.println("Positive");
} else {
    System.out.println("Non-positive");
}

// Transformed
if (!(sum > 0)) {
    System.out.println("Non-positive");
} else {
    System.out.println("Positive");
}
```

### 3. Mathematical Properties
**Commutativity**
```java
// Original
result = a + b * c;

// Transformed
result = b * c + a;
```

**Identity Operations**
```java
// Original
result = value + 0;

// Transformed
result = value;
```

**Associativity**
```java
// Original
result = (a + b) + c;

// Transformed
result = a + (b + c);
```

**Strength Reduction**
```java
// Original
result = x * 2;

// Transformed
result = x << 1;
```

### 4. De Morgan's Laws
```java
// Original
if (!(a && b)) {
    // code
}

// Transformed
if (!a || !b) {
    // code
}
```

### 5. Ternary ↔ If-Else Conversions
```java
// Original ternary
String message = condition ? "yes" : "no";

// Transformed to if-else
String message;
if (condition) {
    message = "yes";
} else {
    message = "no";
}
```

### 6. Switch ↔ If-Else Chain Conversions
```java
// Original switch
switch (value) {
    case 1:
        return "one";
    case 2:
        return "two";
    default:
        return "other";
}

// Transformed to if-else chain
if (value == 1) {
    return "one";
} else if (value == 2) {
    return "two";
} else {
    return "other";
}
```

### 7. Variable Operations
**Variable Inlining**
```java
// Original
int temp = a + b;
return temp * 2;

// Transformed
return (a + b) * 2;
```

## Usage

### Command Line Interface
```bash
# Generate semantic augmentations
python semantic_augment_slices.py \
    --slices_dir /path/to/slices \
    --out_dir /path/to/augmented \
    --variants_per_file 50 \
    --seed 42
```

### Integration with Pipeline
The semantic augmentation system is integrated into the existing pipelines:

1. **Annotation Type Pipeline** (`annotation_type_pipeline.py`)
2. **Simple Annotation Type Pipeline** (`simple_annotation_type_pipeline.py`)

Both pipelines now use semantic transformations by default.

### Test the System
```bash
# Run the test script
python test_semantic_augmentation.py
```

## Benefits

### 1. Slicer Resistance
- **Semantic Preservation**: Transformations maintain original meaning
- **Structural Changes**: Alters syntax without changing behavior
- **Slicer Survival**: Less likely to be removed by program slicers

### 2. Training Quality
- **Meaningful Variations**: Each variant provides different syntactic patterns
- **Consistent Semantics**: Models learn from semantically equivalent code
- **Better Generalization**: Exposure to diverse syntactic structures

### 3. Reduced Noise
- **No Random Code**: Eliminates irrelevant random code injection
- **Focused Transformations**: Only applies meaningful changes
- **Cleaner Training Data**: Higher quality augmented examples

## Configuration

### Transformation Probabilities
The system applies transformations with configurable probabilities:
- **Loop conversions**: 50% chance
- **Guard reversals**: 40% chance  
- **Mathematical properties**: 20-40% chance per type
- **Logical transformations**: 40% chance
- **Control flow conversions**: 50% chance
- **Variable operations**: 30% chance

### Variants per File
- **Default**: 50 variants per file
- **Rationale**: Semantic transformations are more meaningful than random code
- **Configurable**: Can be adjusted based on needs

## Implementation Details

### Core Classes
- **`SemanticTransformer`**: Main transformation engine
- **Transformation Methods**: Individual transformation functions
- **Pattern Matching**: Regex-based code pattern recognition
- **Safety Checks**: Ensures transformations don't break code

### Error Handling
- **Graceful Degradation**: Continues if individual transformations fail
- **Validation**: Ensures transformed code remains compilable
- **Logging**: Comprehensive logging of transformation results

## Comparison with Previous System

### Old System (Random Augmentation)
- ❌ Injected random, irrelevant code
- ❌ Often removed by slicers
- ❌ Added noise to training data
- ❌ Poor semantic coherence

### New System (Semantic Augmentation)
- ✅ Applies semantic-preserving transformations
- ✅ More likely to survive slicing
- ✅ Provides meaningful variations
- ✅ Maintains semantic coherence

## Future Enhancements

### Planned Improvements
1. **More Transformations**: Additional semantic-preserving patterns
2. **Language Support**: Extend to other programming languages
3. **Context Awareness**: Apply transformations based on code context
4. **Quality Metrics**: Measure transformation effectiveness

### Research Directions
1. **Slicer Analysis**: Study which transformations survive different slicers
2. **Model Performance**: Measure impact on training effectiveness
3. **Transformation Selection**: Learn which transformations are most valuable
4. **Automated Discovery**: Find new semantic-preserving patterns

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure `semantic_augment_slices.py` is in the path
2. **Memory Usage**: Large numbers of variants may consume significant memory
3. **Transformation Failures**: Some complex code patterns may not transform

### Debugging
```bash
# Test with a single file
python semantic_augment_slices.py \
    --slices_dir /path/to/single/file \
    --out_dir /path/to/output \
    --variants_per_file 5

# Check transformation results
ls -la /path/to/output/
```

## Conclusion

The semantic augmentation system represents a significant improvement over random code injection. By focusing on semantic-preserving transformations that slicers are less likely to remove, it provides higher quality training data for the GenDATA machine learning models. This approach aligns with the project's goal of creating robust, slicer-resistant training data for Checker Framework annotation placement.