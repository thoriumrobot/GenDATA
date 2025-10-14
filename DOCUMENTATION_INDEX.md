# GenDATA Documentation Index

## Current Status: Eclipse JDT Implementation Complete ✅

All regex-based parsing has been successfully replaced with robust Eclipse JDT AST parsing. The system now provides 100% accurate Java code analysis and transformation capabilities.

## Primary Documentation

### 🎉 **JDT Implementation**
- **[JDT_IMPLEMENTATION_COMPLETE.md](JDT_IMPLEMENTATION_COMPLETE.md)** - Complete documentation of the Eclipse JDT implementation
- **[JDT_MIGRATION_SUMMARY.md](JDT_MIGRATION_SUMMARY.md)** - Summary of the migration from regex to JDT

### 📚 **System Documentation**
- **[README.md](README.md)** - Main project overview and features
- **[README_OPTIMIZED.md](README_OPTIMIZED.md)** - Optimized pipeline documentation
- **[UPDATED_PIPELINE_DOCUMENTATION.md](UPDATED_PIPELINE_DOCUMENTATION.md)** - Current pipeline status
- **[FINAL_IMPLEMENTATION_SUMMARY.md](FINAL_IMPLEMENTATION_SUMMARY.md)** - Implementation summary

### 🔧 **Technical Guides**
- **[AUGMENT_FIRST_GUIDE.md](AUGMENT_FIRST_GUIDE.md)** - Augment-first pipeline guide
- **[RANDOM_WALK_OPTIMIZATION_GUIDE.md](RANDOM_WALK_OPTIMIZATION_GUIDE.md)** - Random walk optimization
- **[ENHANCED_SOOT_SLICER_GUIDE.md](ENHANCED_SOOT_SLICER_GUIDE.md)** - Enhanced Soot slicer
- **[BALANCED_TRAINING_GUIDE.md](BALANCED_TRAINING_GUIDE.md)** - Balanced training system
- **[ANNOTATION_TYPE_MODELS_GUIDE.md](ANNOTATION_TYPE_MODELS_GUIDE.md)** - Annotation-specific models

### 📊 **Evaluation and Results**
- **[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)** - Evaluation system guide
- **[ABLATION_STUDY_RESULTS.md](ABLATION_STUDY_RESULTS.md)** - Ablation study results

## Key Features

### ✅ **Eclipse JDT Integration**
- 100% AST-based Java parsing
- 27 semantic transformations with AST rewriting
- Robust error handling and validation
- Full compatibility with random walk optimization

### ✅ **Enhanced Pipeline**
- Lower Bound Checker integration
- Warning-based slicing with Soot
- Intelligent model selection
- CFG generation for ML models

### ✅ **Random Walk Optimization**
- RL, MCTS, Graph-based, and Evolutionary methods
- Optimal augmentation structure discovery
- Full JDT compatibility verified

### ✅ **Performance**
- GPU acceleration (NVIDIA GeForce RTX 4070 Ti SUPER)
- Balanced training system
- Comprehensive test coverage

## Quick Start

1. **Build JDT Services:**
   ```bash
   ./gradlew jdtParserJar jdtTransformerJar
   ```

2. **Run Tests:**
   ```bash
   ./gradlew test
   python3 -m unittest test_jdt_service
   ```

3. **Use JDT Services:**
   ```python
   from jdt_service import JdtParserService
   from jdt_semantic_transformer import JdtSemanticTransformer
   
   # Parse code locations
   service = JdtParserService()
   locations = service.parse_code_locations_from_string(java_code)
   
   # Apply transformations
   transformer = JdtSemanticTransformer(seed=42)
   result = transformer.transform_code(java_code, ['guard_reversal'], 'enhanced')
   ```

## Architecture Overview

```
Python Layer (Wrappers)
    ↓
Java Services (JDT-based)
    ↓
Eclipse JDT Core (AST Parsing)
```

The system uses a clean separation between Python wrappers and Java services, maintaining the existing subprocess integration pattern for robustness while providing accurate AST-based parsing and transformation capabilities.

## Status: Production Ready ✅

All components have been thoroughly tested and are ready for production use. The JDT implementation provides significant improvements in accuracy and reliability while maintaining full compatibility with existing systems.
