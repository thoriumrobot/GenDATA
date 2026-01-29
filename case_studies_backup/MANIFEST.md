# Case Studies Backup Manifest

**Created**: 2026-01-24 19:07:39

## Purpose

This directory contains backups of all case study projects used for training
and evaluating annotation placement models for the Checker Framework.

## Projects by Category

### Annotated Projects (with injected annotations for training)
- ✓ `commons-dbutils/`
- ✓ `kryo/`
- ✓ `guice/`

### Training Examples (hand-crafted examples)
- ✓ `training_sql_quotes/`
- ✓ `training_signature/`

### Checker Framework Test Suites
- ✓ `cf_sqlquotes_tests/`
- ✓ `cf_signature_tests/`

### Evaluation Projects
- ✓ `agrona/`
- ✓ `eclipse-collections/`
- ✓ `guava/`
- ✓ `hipparchus/`
- ✓ `jfreechart/`
- ✓ `plume-lib/`

### Other Projects
- ✓ `cglib/`
- ✓ `commons-dbcp/`
- ✓ `mybatis-3/`

## Backup Summary

- **Successful**: 16
- **Failed**: 0
- **Skipped**: 0

## Checkers Covered

- **Lower Bound Checker**: `@Positive`, `@NonNegative`, `@GTENegativeOne`
- **SQL Quotes Checker**: `@SqlEvenQuotes`, `@SqlOddQuotes`
- **Signature String Checker**: `@BinaryName`, `@FullyQualifiedName`, `@FieldDescriptor`, `@ClassGetName`, `@InternalForm`

## Usage

To restore a project:
```bash
cp -r case_studies_backup/PROJECT_NAME case_studies/
```

To restore all projects:
```bash
cp -r case_studies_backup/* case_studies/
```
