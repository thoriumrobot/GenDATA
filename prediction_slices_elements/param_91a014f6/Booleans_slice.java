// Source-based slice around line 166
// Method: <com.google.common.primitives.Booleans: int indexOf(boolean[],boolean,int,int)>

   * @param target a primitive {@code boolean} value
   * @return the least index {@code i} for which {@code array[i] == target}, or {@code -1} if no
   *     such index exists.
   */
  public static int indexOf(boolean[] array, boolean target) {
    return indexOf(array, target, 0, array.length);
  }

  // TODO(kevinb): consider making this public
  private static int indexOf(boolean[] array, boolean target, int start, int end) {
    for (int i = start; i < end; i++) {
      if (array[i] == target) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Returns the start position of the first occurrence of the specified {@code target} within
