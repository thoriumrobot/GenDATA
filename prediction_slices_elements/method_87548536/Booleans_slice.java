// Source-based slice around line 161
// Method: <com.google.common.primitives.Booleans: int indexOf(boolean[],boolean)>

   *
   * <p><b>Note:</b> consider representing the array as a {@link java.util.BitSet} instead, and
   * using {@link java.util.BitSet#nextSetBit(int)} or {@link java.util.BitSet#nextClearBit(int)}.
   *
   * @param array an array of {@code boolean} values, possibly empty
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
