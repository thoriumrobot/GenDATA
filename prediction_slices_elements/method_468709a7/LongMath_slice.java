// Source-based slice around line 78
// Method: <com.google.common.math.LongMath: long floorPowerOfTwo(long)>

  }

  /**
   * Returns the largest power of two less than or equal to {@code x}. This is equivalent to {@code
   * checkedPow(2, log2(x, FLOOR))}.
   *
   * @throws IllegalArgumentException if {@code x <= 0}
   * @since 20.0
   */
  public static long floorPowerOfTwo(long x) {
    checkPositive("x", x);

    // Long.highestOneBit was buggy on GWT.  We've fixed it, but I'm not certain when the fix will
    // be released.
    return 1L << ((Long.SIZE - 1) - Long.numberOfLeadingZeros(x));
  }

  /**
   * Returns {@code true} if {@code x} represents a power of two.
   *
