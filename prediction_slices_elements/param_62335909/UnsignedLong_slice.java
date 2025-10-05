// Source-based slice around line 108
// Method: <com.google.common.primitives.UnsignedLong: UnsignedLong valueOf(String)>


  /**
   * Returns an {@code UnsignedLong} holding the value of the specified {@code String}, parsed as an
   * unsigned {@code long} value.
   *
   * @throws NumberFormatException if the string does not contain a parsable unsigned {@code long}
   *     value
   */
  @CanIgnoreReturnValue
  public static UnsignedLong valueOf(String string) {
    return valueOf(string, 10);
  }

  /**
   * Returns an {@code UnsignedLong} holding the value of the specified {@code String}, parsed as an
   * unsigned {@code long} value in the specified radix.
   *
   * @throws NumberFormatException if the string does not contain a parsable unsigned {@code long}
   *     value, or {@code radix} is not between {@link Character#MIN_RADIX} and {@link
   *     Character#MAX_RADIX}
