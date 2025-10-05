// Source-based slice around line 404
// Method: <com.google.common.primitives.ImmutableDoubleArray: int lastIndexOf(double)>

    }
    return -1;
  }

  /**
   * Returns the largest index for which {@link #get} returns {@code target}, or {@code -1} if no
   * such index exists. Values are compared as if by {@link Double#equals}. Equivalent to {@code
   * asList().lastIndexOf(target)}.
   */
  public int lastIndexOf(double target) {
    for (int i = end - 1; i >= start; i--) {
      if (areEqual(array[i], target)) {
        return i - start;
      }
    }
    return -1;
  }

  /**
   * Returns {@code true} if {@code target} is present at any index in this array. Values are
