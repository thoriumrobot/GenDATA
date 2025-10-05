// Source-based slice around line 390
// Method: <com.google.common.primitives.ImmutableDoubleArray: int indexOf(double)>

    Preconditions.checkElementIndex(index, length());
    return array[start + index];
  }

  /**
   * Returns the smallest index for which {@link #get} returns {@code target}, or {@code -1} if no
   * such index exists. Values are compared as if by {@link Double#equals}. Equivalent to {@code
   * asList().indexOf(target)}.
   */
  public int indexOf(double target) {
    for (int i = start; i < end; i++) {
      if (areEqual(array[i], target)) {
        return i - start;
      }
    }
    return -1;
  }

  /**
   * Returns the largest index for which {@link #get} returns {@code target}, or {@code -1} if no
