// Source-based slice around line 68
// Method: com.google.common.primitives.Ints.MAX_POWER_OF_TWO

  // The constants value gets inlined here.
  @SuppressWarnings("AndroidJdkLibsChecker")
  public static final int BYTES = Integer.BYTES;

  /**
   * The largest power of two that can be represented as an {@code int}.
   *
   * @since 10.0
   */
  public static final int MAX_POWER_OF_TWO = 1 << (Integer.SIZE - 2);

  /**
   * Returns a hash code for {@code value}; obsolete alternative to {@link Integer#hashCode(int)}.
   *
   * @param value a primitive {@code int} value
   * @return a hash code for the value
   */
  @InlineMe(replacement = "Integer.hashCode(value)")
  @InlineMeValidationDisabled(
      "The hash code of a int is the int itself, so it's simplest to return that.")
