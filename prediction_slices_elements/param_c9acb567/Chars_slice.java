// Source-based slice around line 201
// Method: <com.google.common.primitives.Chars: int lastIndexOf(char[],char)>


  /**
   * Returns the index of the last appearance of the value {@code target} in {@code array}.
   *
   * @param array an array of {@code char} values, possibly empty
   * @param target a primitive {@code char} value
   * @return the greatest index {@code i} for which {@code array[i] == target}, or {@code -1} if no
   *     such index exists.
   */
  public static int lastIndexOf(char[] array, char target) {
    return lastIndexOf(array, target, 0, array.length);
  }

  // TODO(kevinb): consider making this public
  private static int lastIndexOf(char[] array, char target, int start, int end) {
    for (int i = end - 1; i >= start; i--) {
      if (array[i] == target) {
        return i;
      }
    }
