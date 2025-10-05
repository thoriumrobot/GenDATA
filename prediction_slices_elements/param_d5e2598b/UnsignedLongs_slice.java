// Source-based slice around line 340
// Method: <com.google.common.primitives.UnsignedLongs: long parseUnsignedLong(String,int)>

   * @param string the string containing the unsigned {@code long} representation to be parsed.
   * @param radix the radix to use while parsing {@code string}
   * @throws NumberFormatException if the string does not contain a valid unsigned {@code long} with
   *     the given radix, or if {@code radix} is not between {@link Character#MIN_RADIX} and {@link
   *     Character#MAX_RADIX}.
   * @throws NullPointerException if {@code string} is null (in contrast to {@link
   *     Long#parseLong(String)})
   */
  @CanIgnoreReturnValue
  public static long parseUnsignedLong(String string, int radix) {
    checkNotNull(string);
    if (string.length() == 0) {
      throw new NumberFormatException("empty string");
    }
    if (radix < Character.MIN_RADIX || radix > Character.MAX_RADIX) {
      throw new NumberFormatException("illegal radix: " + radix);
    }

    int maxSafePos = ParseOverflowDetection.maxSafeDigits[radix] - 1;
    long value = 0;
