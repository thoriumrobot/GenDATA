// Source-based slice around line 78
// Method: <com.google.common.math.IntMath: int floorPowerOfTwo(int)>

  }

  /**
   * Returns the largest power of two less than or equal to {@code x}. This is equivalent to {@code
   * checkedPow(2, log2(x, FLOOR))}.
   *
   * @throws IllegalArgumentException if {@code x <= 0}
   * @since 20.0
   */
  public static int floorPowerOfTwo(int x) {
    checkPositive("x", x);
    return Integer.highestOneBit(x);
  }

  /**
   * Returns {@code true} if {@code x} represents a power of two.
   *
   * <p>This differs from {@code Integer.bitCount(x) == 1}, because {@code
   * Integer.bitCount(Integer.MIN_VALUE) == 1}, but {@link Integer#MIN_VALUE} is not a power of two.
   */
