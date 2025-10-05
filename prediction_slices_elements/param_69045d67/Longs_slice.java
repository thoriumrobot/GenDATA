// Source-based slice around line 104
// Method: <com.google.common.primitives.Longs: boolean contains(long[],long)>

  }

  /**
   * Returns {@code true} if {@code target} is present as an element anywhere in {@code array}.
   *
   * @param array an array of {@code long} values, possibly empty
   * @param target a primitive {@code long} value
   * @return {@code true} if {@code array[i] == target} for some value of {@code i}
   */
  public static boolean contains(long[] array, long target) {
    for (long value : array) {
      if (value == target) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns the index of the first appearance of the value {@code target} in {@code array}.
