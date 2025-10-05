// Source-based slice around line 131
// Method: <com.google.common.collect.EnumMultiset: int count(Object)>

    return distinctElements;
  }

  @Override
  public int size() {
    return Ints.saturatedCast(size);
  }

  @Override
  public int count(@Nullable Object element) {
    // isActuallyE checks for null, but we check explicitly to help nullness checkers.
    if (element == null || !isActuallyE(element)) {
      return 0;
    }
    Enum<?> e = (Enum<?>) element;
    return counts[e.ordinal()];
  }

  // Modification Operations
  @CanIgnoreReturnValue
