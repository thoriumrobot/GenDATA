// Source-based slice around line 185
// Method: <com.google.common.primitives.Floats: int lastIndexOf(float[],float)>

  /**
   * Returns the index of the last appearance of the value {@code target} in {@code array}. Note
   * that this always returns {@code -1} when {@code target} is {@code NaN}.
   *
   * @param array an array of {@code float} values, possibly empty
   * @param target a primitive {@code float} value
   * @return the greatest index {@code i} for which {@code array[i] == target}, or {@code -1} if no
   *     such index exists.
   */
  public static int lastIndexOf(float[] array, float target) {
    return lastIndexOf(array, target, 0, array.length);
  }

  // TODO(kevinb): consider making this public
  private static int lastIndexOf(float[] array, float target, int start, int end) {
    for (int i = end - 1; i >= start; i--) {
      if (array[i] == target) {
        return i;
      }
    }
