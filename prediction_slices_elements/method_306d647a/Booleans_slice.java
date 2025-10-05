// Source-based slice around line 141
// Method: <com.google.common.primitives.Booleans: boolean contains(boolean[],boolean)>

   *
   * <p><b>Note:</b> consider representing the array as a {@link java.util.BitSet} instead,
   * replacing {@code Booleans.contains(array, true)} with {@code !bitSet.isEmpty()} and {@code
   * Booleans.contains(array, false)} with {@code bitSet.nextClearBit(0) == sizeOfBitSet}.
   *
   * @param array an array of {@code boolean} values, possibly empty
   * @param target a primitive {@code boolean} value
   * @return {@code true} if {@code array[i] == target} for some value of {@code i}
   */
  public static boolean contains(boolean[] array, boolean target) {
    for (boolean value : array) {
      if (value == target) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns the index of the first appearance of the value {@code target} in {@code array}.
