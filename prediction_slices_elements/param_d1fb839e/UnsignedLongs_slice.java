// Source-based slice around line 322
// Method: <com.google.common.primitives.UnsignedLongs: long parseUnsignedLong(String)>

   *
   * <p><b>Java 8+ users:</b> use {@link Long#parseUnsignedLong(String)} instead.
   *
   * @throws NumberFormatException if the string does not contain a valid unsigned {@code long}
   *     value
   * @throws NullPointerException if {@code string} is null (in contrast to {@link
   *     Long#parseLong(String)})
   */
  @CanIgnoreReturnValue
  public static long parseUnsignedLong(String string) {
    return parseUnsignedLong(string, 10);
  }

  /**
   * Returns the unsigned {@code long} value represented by a string with the given radix.
   *
   * <p><b>Java 8+ users:</b> use {@link Long#parseUnsignedLong(String, int)} instead.
   *
   * @param string the string containing the unsigned {@code long} representation to be parsed.
   * @param radix the radix to use while parsing {@code string}
