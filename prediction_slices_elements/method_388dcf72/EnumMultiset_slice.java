// Source-based slice around line 143
// Method: <com.google.common.collect.EnumMultiset: int add(E,int)>

      return 0;
    }
    Enum<?> e = (Enum<?>) element;
    return counts[e.ordinal()];
  }

  // Modification Operations
  @CanIgnoreReturnValue
  @Override
  public int add(E element, int occurrences) {
    checkIsE(element);
    checkNonnegative(occurrences, "occurrences");
    if (occurrences == 0) {
      return count(element);
    }
    int index = element.ordinal();
    int oldCount = counts[index];
    long newCount = (long) oldCount + occurrences;
    checkArgument(newCount <= Integer.MAX_VALUE, "too many occurrences: %s", newCount);
    counts[index] = (int) newCount;
