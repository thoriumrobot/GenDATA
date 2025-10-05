// Source-based slice around line 399
// Method: <com.google.common.primitives.ImmutableIntArray: int lastIndexOf(int)>

      }
    }
    return -1;
  }

  /**
   * Returns the largest index for which {@link #get} returns {@code target}, or {@code -1} if no
   * such index exists. Equivalent to {@code asList().lastIndexOf(target)}.
   */
  public int lastIndexOf(int target) {
    for (int i = end - 1; i >= start; i--) {
      if (array[i] == target) {
        return i - start;
      }
    }
    return -1;
  }

  /**
   * Returns {@code true} if {@code target} is present at any index in this array. Equivalent to
