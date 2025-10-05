// Source-based slice around line 63
// Method: com.google.common.primitives.Doubles.BYTES

  /**
   * The number of bytes required to represent a primitive {@code double} value.
   *
   * <p>Prefer {@link Double#BYTES} instead.
   *
   * @since 10.0
   */
  // The constants value gets inlined here.
  @SuppressWarnings("AndroidJdkLibsChecker")
  public static final int BYTES = Double.BYTES;

  /**
   * Returns a hash code for {@code value}; obsolete alternative to {@link Double#hashCode(double)}.
   *
   * @param value a primitive {@code double} value
   * @return a hash code for the value
   */
  @InlineMe(replacement = "Double.hashCode(value)")
  public static int hashCode(double value) {
    return Double.hashCode(value);
