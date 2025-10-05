// Source-based slice around line 134
// Method: <com.google.common.primitives.Doubles: int indexOf(double[],double)>

  /**
   * Returns the index of the first appearance of the value {@code target} in {@code array}. Note
   * that this always returns {@code -1} when {@code target} is {@code NaN}.
   *
   * @param array an array of {@code double} values, possibly empty
   * @param target a primitive {@code double} value
   * @return the least index {@code i} for which {@code array[i] == target}, or {@code -1} if no
   *     such index exists.
   */
  public static int indexOf(double[] array, double target) {
    return indexOf(array, target, 0, array.length);
  }

  // TODO(kevinb): consider making this public
  private static int indexOf(double[] array, double target, int start, int end) {
    for (int i = start; i < end; i++) {
      if (array[i] == target) {
        return i;
      }
    }
