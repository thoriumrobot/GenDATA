// Source-based slice around line 459
// Method: <com.google.common.primitives.UnsignedLongs: String toString(long,int)>

   * unsigned.
   *
   * <p><b>Java 8+ users:</b> use {@link Long#toUnsignedString(long, int)} instead.
   *
   * @param x the value to convert to a string.
   * @param radix the radix to use while working with {@code x}
   * @throws IllegalArgumentException if {@code radix} is not between {@link Character#MIN_RADIX}
   *     and {@link Character#MAX_RADIX}.
   */
  public static String toString(long x, int radix) {
    checkArgument(
        radix >= Character.MIN_RADIX && radix <= Character.MAX_RADIX,
        "radix (%s) must be between Character.MIN_RADIX and Character.MAX_RADIX",
        radix);
    if (x == 0) {
      // Simply return "0"
      return "0";
    } else if (x > 0) {
      return Long.toString(x, radix);
    } else {
