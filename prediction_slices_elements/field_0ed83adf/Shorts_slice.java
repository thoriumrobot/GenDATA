// Source-based slice around line 66
// Method: com.google.common.primitives.Shorts.MAX_POWER_OF_TWO

  // The constants value gets inlined here.
  @SuppressWarnings("AndroidJdkLibsChecker")
  public static final int BYTES = Short.BYTES;

  /**
   * The largest power of two that can be represented as a {@code short}.
   *
   * @since 10.0
   */
  public static final short MAX_POWER_OF_TWO = 1 << (Short.SIZE - 2);

  /**
   * Returns a hash code for {@code value}; obsolete alternative to {@link Short#hashCode(short)}.
   *
   * @param value a primitive {@code short} value
   * @return a hash code for the value
   */
  @InlineMe(replacement = "Short.hashCode(value)")
  @InlineMeValidationDisabled(
      "The hash code of a short is the int version of the short itself, so it's simplest to return"
