// Source-based slice around line 160
// Method: <com.google.common.primitives.Ints: int indexOf(int[],int)>


  /**
   * Returns the index of the first appearance of the value {@code target} in {@code array}.
   *
   * @param array an array of {@code int} values, possibly empty
   * @param target a primitive {@code int} value
   * @return the least index {@code i} for which {@code array[i] == target}, or {@code -1} if no
   *     such index exists.
   */
  public static int indexOf(int[] array, int target) {
    return indexOf(array, target, 0, array.length);
  }

  // TODO(kevinb): consider making this public
  private static int indexOf(int[] array, int target, int start, int end) {
    for (int i = start; i < end; i++) {
      if (array[i] == target) {
        return i;
      }
    }
