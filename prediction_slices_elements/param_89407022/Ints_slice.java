// Source-based slice around line 216
// Method: <com.google.common.primitives.Ints: int lastIndexOf(int[],int,int,int)>

   * @param target a primitive {@code int} value
   * @return the greatest index {@code i} for which {@code array[i] == target}, or {@code -1} if no
   *     such index exists.
   */
  public static int lastIndexOf(int[] array, int target) {
    return lastIndexOf(array, target, 0, array.length);
  }

  // TODO(kevinb): consider making this public
  private static int lastIndexOf(int[] array, int target, int start, int end) {
    for (int i = end - 1; i >= start; i--) {
      if (array[i] == target) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Returns the least value present in {@code array}.
