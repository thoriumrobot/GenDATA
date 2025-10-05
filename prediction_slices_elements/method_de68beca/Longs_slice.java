// Source-based slice around line 76
// Method: <com.google.common.primitives.Longs: int hashCode(long)>

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
  }

  /**
   * Compares the two specified {@code long} values. The sign of the value returned is the same as
   * that of {@code ((Long) a).compareTo(b)}.
   *
   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use the
   * equivalent {@link Long#compare} method instead.
   *
