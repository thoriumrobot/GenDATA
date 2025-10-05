// Source-based slice around line 126
// Method: <com.google.common.primitives.Booleans: int compare(boolean,boolean)>

   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use the
   * equivalent {@link Boolean#compare} method instead.
   *
   * @param a the first {@code boolean} to compare
   * @param b the second {@code boolean} to compare
   * @return a positive number if only {@code a} is {@code true}, a negative number if only {@code
   *     b} is true, or zero if {@code a == b}
   */
  @InlineMe(replacement = "Boolean.compare(a, b)")
  public static int compare(boolean a, boolean b) {
    return Boolean.compare(a, b);
  }

  /**
   * Returns {@code true} if {@code target} is present as an element anywhere in {@code array}.
   *
   * <p><b>Note:</b> consider representing the array as a {@link java.util.BitSet} instead,
   * replacing {@code Booleans.contains(array, true)} with {@code !bitSet.isEmpty()} and {@code
   * Booleans.contains(array, false)} with {@code bitSet.nextClearBit(0) == sizeOfBitSet}.
   *
