// Source-based slice around line 137
// Method: <com.google.common.primitives.Shorts: boolean contains(short[],short)>

  }

  /**
   * Returns {@code true} if {@code target} is present as an element anywhere in {@code array}.
   *
   * @param array an array of {@code short} values, possibly empty
   * @param target a primitive {@code short} value
   * @return {@code true} if {@code array[i] == target} for some value of {@code i}
   */
  public static boolean contains(short[] array, short target) {
    for (short value : array) {
      if (value == target) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns the index of the first appearance of the value {@code target} in {@code array}.
