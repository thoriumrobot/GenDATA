// Source-based slice around line 724
// Method: <com.google.common.primitives.Floats: Float tryParse(String)>

   * are expected.
   *
   * @param string the string representation of a {@code float} value
   * @return the floating point value represented by {@code string}, or {@code null} if {@code
   *     string} has a length of zero or cannot be parsed as a {@code float} value
   * @throws NullPointerException if {@code string} is {@code null}
   * @since 14.0
   */
  @GwtIncompatible // regular expressions
  public static @Nullable Float tryParse(String string) {
    if (Doubles.FLOATING_POINT_PATTERN.matcher(string).matches()) {
      // TODO(lowasser): could be potentially optimized, but only with
      // extensive testing
      try {
        return Float.parseFloat(string);
      } catch (NumberFormatException e) {
        // Float.parseFloat has changed specs several times, so fall through
        // gracefully
      }
    }
