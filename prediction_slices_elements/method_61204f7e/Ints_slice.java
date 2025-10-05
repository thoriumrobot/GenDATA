// Source-based slice around line 840
// Method: <com.google.common.primitives.Ints: Integer tryParse(String,int)>

   * @param string the string representation of an integer value
   * @param radix the radix to use when parsing
   * @return the integer value represented by {@code string} using {@code radix}, or {@code null} if
   *     {@code string} has a length of zero or cannot be parsed as an integer value
   * @throws IllegalArgumentException if {@code radix < Character.MIN_RADIX} or {@code radix >
   *     Character.MAX_RADIX}
   * @throws NullPointerException if {@code string} is {@code null}
   * @since 19.0
   */
  public static @Nullable Integer tryParse(String string, int radix) {
    Long result = Longs.tryParse(string, radix);
    if (result == null || result.longValue() != result.intValue()) {
      return null;
    } else {
      return result.intValue();
    }
  }
}
