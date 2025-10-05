// Source-based slice around line 376
// Method: <com.google.common.primitives.Longs: Long tryParse(String)>

   * <p>Note that strings prefixed with ASCII {@code '+'} are rejected, even though {@link
   * Integer#parseInt(String)} accepts them.
   *
   * @param string the string representation of a long value
   * @return the long value represented by {@code string}, or {@code null} if {@code string} has a
   *     length of zero or cannot be parsed as a long value
   * @throws NullPointerException if {@code string} is {@code null}
   * @since 14.0
   */
  public static @Nullable Long tryParse(String string) {
    return tryParse(string, 10);
  }

  /**
   * Parses the specified string as a signed long value using the specified radix. The ASCII
   * character {@code '-'} (<code>'&#92;u002D'</code>) is recognized as the minus sign.
   *
   * <p>Unlike {@link Long#parseLong(String, int)}, this method returns {@code null} instead of
   * throwing an exception if parsing fails. Additionally, this method only accepts ASCII digits,
   * and returns {@code null} if non-ASCII digits are present in the string.
