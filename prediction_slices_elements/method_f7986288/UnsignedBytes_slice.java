// Source-based slice around line 213
// Method: <com.google.common.primitives.UnsignedBytes: byte parseUnsignedByte(String)>

   * Returns the unsigned {@code byte} value represented by the given decimal string.
   *
   * @throws NumberFormatException if the string does not contain a valid unsigned {@code byte}
   *     value
   * @throws NullPointerException if {@code string} is null (in contrast to {@link
   *     Byte#parseByte(String)})
   * @since 13.0
   */
  @CanIgnoreReturnValue
  public static byte parseUnsignedByte(String string) {
    return parseUnsignedByte(string, 10);
  }

  /**
   * Returns the unsigned {@code byte} value represented by a string with the given radix.
   *
   * @param string the string containing the unsigned {@code byte} representation to be parsed.
   * @param radix the radix to use while parsing {@code string}
   * @throws NumberFormatException if the string does not contain a valid unsigned {@code byte} with
   *     the given radix, or if {@code radix} is not between {@link Character#MIN_RADIX} and {@link
