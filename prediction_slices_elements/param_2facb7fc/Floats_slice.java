// Source-based slice around line 113
// Method: <com.google.common.primitives.Floats: boolean contains(float[],float)>


  /**
   * Returns {@code true} if {@code target} is present as an element anywhere in {@code array}. Note
   * that this always returns {@code false} when {@code target} is {@code NaN}.
   *
   * @param array an array of {@code float} values, possibly empty
   * @param target a primitive {@code float} value
   * @return {@code true} if {@code array[i] == target} for some value of {@code i}
   */
  public static boolean contains(float[] array, float target) {
    for (float value : array) {
      if (value == target) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns the index of the first appearance of the value {@code target} in {@code array}. Note
