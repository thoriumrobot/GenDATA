// Source-based slice around line 150
// Method: <com.google.common.primitives.Chars: int indexOf(char[],char)>


  /**
   * Returns the index of the first appearance of the value {@code target} in {@code array}.
   *
   * @param array an array of {@code char} values, possibly empty
   * @param target a primitive {@code char} value
   * @return the least index {@code i} for which {@code array[i] == target}, or {@code -1} if no
   *     such index exists.
   */
  public static int indexOf(char[] array, char target) {
    return indexOf(array, target, 0, array.length);
  }

  // TODO(kevinb): consider making this public
  private static int indexOf(char[] array, char target, int start, int end) {
    for (int i = start; i < end; i++) {
      if (array[i] == target) {
        return i;
      }
    }
