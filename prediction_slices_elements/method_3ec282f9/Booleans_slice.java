// Source-based slice around line 212
// Method: <com.google.common.primitives.Booleans: int lastIndexOf(boolean[],boolean)>


  /**
   * Returns the index of the last appearance of the value {@code target} in {@code array}.
   *
   * @param array an array of {@code boolean} values, possibly empty
   * @param target a primitive {@code boolean} value
   * @return the greatest index {@code i} for which {@code array[i] == target}, or {@code -1} if no
   *     such index exists.
   */
  public static int lastIndexOf(boolean[] array, boolean target) {
    return lastIndexOf(array, target, 0, array.length);
  }

  // TODO(kevinb): consider making this public
  private static int lastIndexOf(boolean[] array, boolean target, int start, int end) {
    for (int i = end - 1; i >= start; i--) {
      if (array[i] == target) {
        return i;
      }
    }
