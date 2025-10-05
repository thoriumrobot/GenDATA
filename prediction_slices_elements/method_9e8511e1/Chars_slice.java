// Source-based slice around line 133
// Method: <com.google.common.primitives.Chars: boolean contains(char[],char)>

  }

  /**
   * Returns {@code true} if {@code target} is present as an element anywhere in {@code array}.
   *
   * @param array an array of {@code char} values, possibly empty
   * @param target a primitive {@code char} value
   * @return {@code true} if {@code array[i] == target} for some value of {@code i}
   */
  public static boolean contains(char[] array, char target) {
    for (char value : array) {
      if (value == target) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns the index of the first appearance of the value {@code target} in {@code array}.
