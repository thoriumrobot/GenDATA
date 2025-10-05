// Source-based slice around line 79
// Method: <com.google.common.primitives.UnsignedInts: long toLong(int)>

  public static int compare(int a, int b) {
    return Ints.compare(flip(a), flip(b));
  }

  /**
   * Returns the value of the given {@code int} as a {@code long}, when treated as unsigned.
   *
   * <p><b>Java 8+ users:</b> use {@link Integer#toUnsignedLong(int)} instead.
   */
  public static long toLong(int value) {
    return value & INT_MASK;
  }

  /**
   * Returns the {@code int} value that, when treated as unsigned, is equal to {@code value}, if
   * possible.
   *
   * @param value a value between 0 and 2<sup>32</sup>-1 inclusive
   * @return the {@code int} value that, when treated as unsigned, equals {@code value}
   * @throws IllegalArgumentException if {@code value} is negative or greater than or equal to
