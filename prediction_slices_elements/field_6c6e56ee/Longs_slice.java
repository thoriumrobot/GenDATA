// Source-based slice around line 67
// Method: com.google.common.primitives.Longs.MAX_POWER_OF_TWO

  // The constants value gets inlined here.
  @SuppressWarnings("AndroidJdkLibsChecker")
  public static final int BYTES = Long.BYTES;

  /**
   * The largest power of two that can be represented as a {@code long}.
   *
   * @since 10.0
   */
  public static final long MAX_POWER_OF_TWO = 1L << (Long.SIZE - 2);

  /**
   * Returns a hash code for {@code value}; obsolete alternative to {@link Long#hashCode(long)}.
   *
   * @param value a primitive {@code long} value
   * @return a hash code for the value
   */
  @InlineMe(replacement = "Long.hashCode(value)")
  public static int hashCode(long value) {
    return Long.hashCode(value);
