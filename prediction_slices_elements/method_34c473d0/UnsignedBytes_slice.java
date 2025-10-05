// Source-based slice around line 194
// Method: <com.google.common.primitives.UnsignedBytes: String toString(byte,int)>

   * Returns a string representation of {@code x} for the given radix, where {@code x} is treated as
   * unsigned.
   *
   * @param x the value to convert to a string.
   * @param radix the radix to use while working with {@code x}
   * @throws IllegalArgumentException if {@code radix} is not between {@link Character#MIN_RADIX}
   *     and {@link Character#MAX_RADIX}.
   * @since 13.0
   */
  public static String toString(byte x, int radix) {
    checkArgument(
        radix >= Character.MIN_RADIX && radix <= Character.MAX_RADIX,
        "radix (%s) must be between Character.MIN_RADIX and Character.MAX_RADIX",
        radix);
    // Benchmarks indicate this is probably not worth optimizing.
    return Integer.toString(toUnsignedInt(x), radix);
  }

  /**
   * Returns the unsigned {@code byte} value represented by the given decimal string.
