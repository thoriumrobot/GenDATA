// Source-based slice around line 128
// Method: com.google.common.math.BigIntegerMath.SQRT2_PRECOMPUTE_THRESHOLD

    }
    throw new AssertionError();
  }

  /*
   * The maximum number of bits in a square root for which we'll precompute an explicit half power
   * of two. This can be any value, but higher values incur more class load time and linearly
   * increasing memory consumption.
   */
  @VisibleForTesting static final int SQRT2_PRECOMPUTE_THRESHOLD = 256;

  @VisibleForTesting
  static final BigInteger SQRT2_PRECOMPUTED_BITS =
      new BigInteger("16a09e667f3bcc908b2fb1366ea957d3e3adec17512775099da2f590b0667322a", 16);

  /**
   * Returns the base-10 logarithm of {@code x}, rounded according to the specified rounding mode.
   *
   * @throws IllegalArgumentException if {@code x <= 0}
   * @throws ArithmeticException if {@code mode} is {@link RoundingMode#UNNECESSARY} and {@code x}
