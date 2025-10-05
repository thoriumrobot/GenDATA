// Source-based slice around line 196
// Method: <com.google.common.collect.RegularImmutableTable: void checkNoDuplicate(R,C,V,V)>

  }

  /**
   * @throws IllegalArgumentException if {@code existingValue} is not null.
   */
  /*
   * We could have declared this method 'static' but the additional compile-time checks achieved by
   * referencing the type variables seem worthwhile.
   */
  final void checkNoDuplicate(R rowKey, C columnKey, @Nullable V existingValue, V newValue) {
    checkArgument(
        existingValue == null,
        "Duplicate key: (row=%s, column=%s), values: [%s, %s].",
        rowKey,
        columnKey,
        newValue,
        existingValue);
  }

  // redeclare to satisfy our test for b/310253115
