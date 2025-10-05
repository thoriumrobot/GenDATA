// Source-based slice around line 209
// Method: com.google.common.math.BigIntegerMath.LN_2

        // Since sqrt(10) is irrational, log10(x) - floorLog can never be exactly 0.5
        BigInteger x2 = x.pow(2);
        BigInteger halfPowerSquared = floorPow.pow(2).multiply(BigInteger.TEN);
        return (x2.compareTo(halfPowerSquared) <= 0) ? floorLog : floorLog + 1;
    }
    throw new AssertionError();
  }

  private static final double LN_10 = Math.log(10);
  private static final double LN_2 = Math.log(2);

  /**
   * Returns the square root of {@code x}, rounded with the specified rounding mode.
   *
   * @throws IllegalArgumentException if {@code x < 0}
   * @throws ArithmeticException if {@code mode} is {@link RoundingMode#UNNECESSARY} and {@code
   *     sqrt(x)} is not an integer
   */
  @GwtIncompatible // TODO
  @SuppressWarnings("fallthrough")
