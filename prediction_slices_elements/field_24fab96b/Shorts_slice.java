// Source-based slice around line 59
// Method: com.google.common.primitives.Shorts.BYTES

  private Shorts() {}

  /**
   * The number of bytes required to represent a primitive {@code short} value.
   *
   * <p>Prefer {@link Short#BYTES} instead.
   */
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
