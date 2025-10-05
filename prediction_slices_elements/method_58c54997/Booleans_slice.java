// Source-based slice around line 217
// Method: <com.google.common.primitives.Booleans: int lastIndexOf(boolean[],boolean,int,int)>

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
    return -1;
  }

  /**
   * Returns the values from each provided array combined into a single array. For example, {@code
