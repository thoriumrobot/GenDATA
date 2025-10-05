// Source-based slice around line 770
// Method: <com.google.common.primitives.Doubles: Double tryParse(String)>

   * are expected.
   *
   * @param string the string representation of a {@code double} value
   * @return the floating point value represented by {@code string}, or {@code null} if {@code
   *     string} has a length of zero or cannot be parsed as a {@code double} value
   * @throws NullPointerException if {@code string} is {@code null}
   * @since 14.0
   */
  @GwtIncompatible // regular expressions
  public static @Nullable Double tryParse(String string) {
    if (FLOATING_POINT_PATTERN.matcher(string).matches()) {
      // TODO(lowasser): could be potentially optimized, but only with
      // extensive testing
      try {
        return Double.parseDouble(string);
      } catch (NumberFormatException e) {
        // Double.parseDouble has changed specs several times, so fall through
        // gracefully
      }
    }
