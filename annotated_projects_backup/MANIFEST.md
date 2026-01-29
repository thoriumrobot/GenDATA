# Annotated Projects Backup Manifest

**Created**: 2026-01-24 17:43:21

## Purpose

This directory contains backups of Java projects that have been annotated with
Checker Framework type annotations for training annotation placement models.

## Directory Structure

```
annotated_projects_backup/
├── sql_quotes/
│   ├── commons-dbutils/     # 51 @SqlEvenQuotes annotations
│   ├── training_sql_quotes/ # Training examples with warnings
│   └── cf_sqlquotes_tests/  # Checker Framework test suite
└── signature_string/
    ├── kryo/                # 3 @BinaryName annotations  
    ├── guice/               # 3 @BinaryName annotations
    ├── training_signature/  # Training examples with warnings
    └── cf_signature_tests/  # Checker Framework test suite
```

## Annotations by Checker

### SQL Quotes Checker
- `@SqlEvenQuotes` - String with even number of SQL quotes
- `@SqlOddQuotes` - String with odd number of SQL quotes
- Import: `org.checkerframework.checker.sqlquotes.qual.*`

### Signature String Checker
- `@BinaryName` - Binary class name (e.g., "java.lang.String")
- `@FullyQualifiedName` - Fully qualified name (e.g., "java.lang.String")  
- `@FieldDescriptor` - Field descriptor (e.g., "Ljava/lang/String;")
- `@ClassGetName` - Result of Class.getName()
- `@InternalForm` - Internal form (e.g., "java/lang/String")
- Import: `org.checkerframework.checker.signature.qual.*`

## Usage

These projects can be used to:
1. Train annotation placement models
2. Test checker warnings before/after annotation
3. Verify warning reduction with correct annotations

## Running Checkers

```bash
# SQL Quotes Checker
/home/ubuntu/checker-framework/checker/bin/javac \
  -processor org.checkerframework.checker.sqlquotes.SqlQuotesChecker \
  YourFile.java

# Signature String Checker
/home/ubuntu/checker-framework/checker/bin/javac \
  -processor org.checkerframework.checker.signature.SignatureChecker \
  YourFile.java
```
