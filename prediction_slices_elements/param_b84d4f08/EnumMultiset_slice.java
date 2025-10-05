// Source-based slice around line 164
// Method: <com.google.common.collect.EnumMultiset: int remove(Object,int)>

      distinctElements++;
    }
    size += occurrences;
    return oldCount;
  }

  // Modification Operations
  @CanIgnoreReturnValue
  @Override
  public int remove(@Nullable Object element, int occurrences) {
    // isActuallyE checks for null, but we check explicitly to help nullness checkers.
    if (element == null || !isActuallyE(element)) {
      return 0;
    }
    Enum<?> e = (Enum<?>) element;
    checkNonnegative(occurrences, "occurrences");
    if (occurrences == 0) {
      return count(element);
    }
    int index = e.ordinal();
