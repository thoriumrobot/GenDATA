// Source-based slice around line 343
// Method: <com.google.common.primitives.UnsignedInts: int parseUnsignedInt(String)>

   * Returns the unsigned {@code int} value represented by the given decimal string.
   *
   * <p><b>Java 8+ users:</b> use {@link Integer#parseUnsignedInt(String)} instead.
   *
   * @throws NumberFormatException if the string does not contain a valid unsigned {@code int} value
   * @throws NullPointerException if {@code s} is null (in contrast to {@link
   *     Integer#parseInt(String)})
   */
  @CanIgnoreReturnValue
  public static int parseUnsignedInt(String s) {
    return parseUnsignedInt(s, 10);
  }

  /**
   * Returns the unsigned {@code int} value represented by a string with the given radix.
   *
   * <p><b>Java 8+ users:</b> use {@link Integer#parseUnsignedInt(String, int)} instead.
   *
   * @param string the string containing the unsigned integer representation to be parsed.
   * @param radix the radix to use while parsing {@code s}; must be between {@link
