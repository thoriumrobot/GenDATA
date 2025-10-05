// Source-based slice around line 121
// Method: <com.google.common.primitives.UnsignedLong: UnsignedLong valueOf(String,int)>

  /**
   * Returns an {@code UnsignedLong} holding the value of the specified {@code String}, parsed as an
   * unsigned {@code long} value in the specified radix.
   *
   * @throws NumberFormatException if the string does not contain a parsable unsigned {@code long}
   *     value, or {@code radix} is not between {@link Character#MIN_RADIX} and {@link
   *     Character#MAX_RADIX}
   */
  @CanIgnoreReturnValue
  public static UnsignedLong valueOf(String string, int radix) {
    return fromLongBits(UnsignedLongs.parseUnsignedLong(string, radix));
  }

  /**
   * Returns the result of adding this and {@code val}. If the result would have more than 64 bits,
   * returns the low 64 bits of the result.
   *
   * @since 14.0
   */
  public UnsignedLong plus(UnsignedLong val) {
