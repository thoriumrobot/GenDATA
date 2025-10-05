// Source-based slice around line 61
// Method: com.google.common.primitives.Ints.BYTES

  private Ints() {}

  /**
   * The number of bytes required to represent a primitive {@code int} value.
   *
   * <p>Prefer {@link Integer#BYTES} instead.
   */
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
