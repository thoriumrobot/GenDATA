// Source-based slice around line 388
// Method: <com.google.common.primitives.ImmutableLongArray: int indexOf(long)>

  public long get(int index) {
    Preconditions.checkElementIndex(index, length());
    return array[start + index];
  }

  /**
   * Returns the smallest index for which {@link #get} returns {@code target}, or {@code -1} if no
   * such index exists. Equivalent to {@code asList().indexOf(target)}.
   */
  public int indexOf(long target) {
    for (int i = start; i < end; i++) {
      if (array[i] == target) {
        return i - start;
      }
    }
    return -1;
  }

  /**
   * Returns the largest index for which {@link #get} returns {@code target}, or {@code -1} if no
