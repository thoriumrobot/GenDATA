# Balanced Training - Quick Reference

**Most Important Files to Understand the System**:

1. **`improved_balanced_dataset_generator.py`** - Core implementation
   - Checker-specific classification (lines 539-701)
   - Checker-specific feature extraction (lines 277-382)

2. **`improved_balanced_annotation_type_trainer.py`** - Training implementation

3. **`train_balanced_sql_quotes_models.py`** - SQL Quotes training script

4. **`train_balanced_signature_string_models.py`** - Signature String training script

5. **`BALANCED_TRAINING_ACCURATE_DOCUMENTATION.md`** - Most accurate documentation (verified facts)

## Current Status (Verified 2025-12-08)

### Datasets
- ✅ SQL Quotes: 2,000 examples, perfect 50/50 balance
- ✅ Signature String: 3,000 examples, perfect 50/50 balance

### Models
- ⚠️ SQL Quotes: 9/14 models with valid stats (64.3% complete)
- ⚠️ Signature String: 11/21 models with valid stats (52.4% complete)

### Performance
- SQL Quotes: 100% accuracy (9 models with valid stats)
- Signature String: 95.18% average accuracy (11 models with valid stats, range 75.50%-100%)

**Important**: Always verify actual model files and training statistics. Some model files exist but lack complete training stats.

## Verification Commands

```bash
# Check dataset balance
cat balanced_datasets_sql_quotes/real_generation_statistics.json
cat balanced_datasets_signature_string/real_generation_statistics.json

# Count model files
ls -1 models_annotation_types_sql_quotes/*_balanced_model.pth | wc -l
ls -1 models_annotation_types_signature_string/*_balanced_model.pth | wc -l

# Verify model stats (see BALANCED_TRAINING_ACCURATE_DOCUMENTATION.md)
```

## Documentation Files

- **`BALANCED_TRAINING_ACCURATE_DOCUMENTATION.md`** - ⭐ **START HERE** - Most accurate, verified information
- `BALANCED_TRAINING_ALL_CHECKERS_DOCUMENTATION.md` - Comprehensive guide (updated with verified info)
- `BALANCED_TRAINING_IMPLEMENTATION_REPORT.md` - Implementation details (updated with verified info)
- `BALANCED_TRAINING_FINAL_REPORT.md` - Summary report (updated with verified info)
- `BALANCED_TRAINING_METRICS_REPORT.md` - Auto-generated metrics (may have inaccuracies, verify model files)

