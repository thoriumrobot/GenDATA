// Source-based slice around line 61
// Method: <com.google.common.primitives.UnsignedLongs: long flip(long)>

  private UnsignedLongs() {}

  public static final long MAX_VALUE = -1L; // Equivalent to 2^64 - 1

  /**
   * A (self-inverse) bijection which converts the ordering on unsigned longs to the ordering on
   * longs, that is, {@code a <= b} as unsigned longs if and only if {@code flip(a) <= flip(b)} as
   * signed longs.
   */
  private static long flip(long a) {
    return a ^ Long.MIN_VALUE;
  }

  /**
   * Compares the two specified {@code long} values, treating them as unsigned values between {@code
   * 0} and {@code 2^64 - 1} inclusive.
   *
   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use the
   * equivalent {@link Long#compareUnsigned(long, long)} method instead.
   *
