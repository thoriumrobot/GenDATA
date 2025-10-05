// Source-based slice around line 391
// Method: <com.google.common.primitives.UnsignedInts: String toString(int,int)>

   * unsigned.
   *
   * <p><b>Java 8+ users:</b> use {@link Integer#toUnsignedString(int, int)} instead.
   *
   * @param x the value to convert to a string.
   * @param radix the radix to use while working with {@code x}
   * @throws IllegalArgumentException if {@code radix} is not between {@link Character#MIN_RADIX}
   *     and {@link Character#MAX_RADIX}.
   */
  public static String toString(int x, int radix) {
    long asLong = x & INT_MASK;
    return Long.toString(asLong, radix);
  }
}
