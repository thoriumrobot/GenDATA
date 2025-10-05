// Source-based slice around line 400
// Method: <com.google.common.primitives.Longs: Long tryParse(String,int)>

   * @param string the string representation of a long value
   * @param radix the radix to use when parsing
   * @return the long value represented by {@code string} using {@code radix}, or {@code null} if
   *     {@code string} has a length of zero or cannot be parsed as a long value
   * @throws IllegalArgumentException if {@code radix < Character.MIN_RADIX} or {@code radix >
   *     Character.MAX_RADIX}
   * @throws NullPointerException if {@code string} is {@code null}
   * @since 19.0
   */
  public static @Nullable Long tryParse(String string, int radix) {
    if (checkNotNull(string).isEmpty()) {
      return null;
    }
    if (radix < Character.MIN_RADIX || radix > Character.MAX_RADIX) {
      throw new IllegalArgumentException(
          "radix must be between MIN_RADIX and MAX_RADIX but was " + radix);
    }
    boolean negative = string.charAt(0) == '-';
    int index = negative ? 1 : 0;
    if (index == string.length()) {
