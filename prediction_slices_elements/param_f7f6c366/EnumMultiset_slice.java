// Source-based slice around line 192
// Method: <com.google.common.collect.EnumMultiset: int setCount(E,int)>

      counts[index] = oldCount - occurrences;
      size -= occurrences;
    }
    return oldCount;
  }

  // Modification Operations
  @CanIgnoreReturnValue
  @Override
  public int setCount(E element, int count) {
    checkIsE(element);
    checkNonnegative(count, "count");
    int index = element.ordinal();
    int oldCount = counts[index];
    counts[index] = count;
    size += count - oldCount;
    if (oldCount == 0 && count > 0) {
      distinctElements++;
    } else if (oldCount > 0 && count == 0) {
      distinctElements--;
