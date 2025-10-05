// Source-based slice around line 60
// Method: com.google.common.primitives.Longs.BYTES

  private Longs() {}

  /**
   * The number of bytes required to represent a primitive {@code long} value.
   *
   * <p>Prefer {@link Long#BYTES} instead.
   */
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
