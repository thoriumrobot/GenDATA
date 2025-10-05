// Source-based slice around line 75
// Method: <com.google.common.primitives.Bytes: boolean contains(byte[],byte)>

  }

  /**
   * Returns {@code true} if {@code target} is present as an element anywhere in {@code array}.
   *
   * @param array an array of {@code byte} values, possibly empty
   * @param target a primitive {@code byte} value
   * @return {@code true} if {@code array[i] == target} for some value of {@code i}
   */
  public static boolean contains(byte[] array, byte target) {
    for (byte value : array) {
      if (value == target) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns the index of the first appearance of the value {@code target} in {@code array}.
