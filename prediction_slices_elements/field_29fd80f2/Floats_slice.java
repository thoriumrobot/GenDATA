// Source-based slice around line 61
// Method: com.google.common.primitives.Floats.BYTES

  /**
   * The number of bytes required to represent a primitive {@code float} value.
   *
   * <p>Prefer {@link Float#BYTES} instead.
   *
   * @since 10.0
   */
  // The constants value gets inlined here.
  @SuppressWarnings("AndroidJdkLibsChecker")
  public static final int BYTES = Float.BYTES;

  /**
   * Returns a hash code for {@code value}; obsolete alternative to {@link Float#hashCode(float)}.
   *
   * @param value a primitive {@code float} value
   * @return a hash code for the value
   */
  @InlineMe(replacement = "Float.hashCode(value)")
  public static int hashCode(float value) {
    return Float.hashCode(value);
