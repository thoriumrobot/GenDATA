// Source-based slice around line 144
// Method: com.google.common.math.IntMath.MAX_POWER_OF_SQRT2_UNSIGNED

        int cmp = MAX_POWER_OF_SQRT2_UNSIGNED >>> leadingZeros;
        // floor(2^(logFloor + 0.5))
        int logFloor = (Integer.SIZE - 1) - leadingZeros;
        return logFloor + lessThanBranchFree(cmp, x);
    }
    throw new AssertionError();
  }

  /** The biggest half power of two that can fit in an unsigned int. */
  @VisibleForTesting static final int MAX_POWER_OF_SQRT2_UNSIGNED = 0xB504F333;

  /**
   * Returns the base-10 logarithm of {@code x}, rounded according to the specified rounding mode.
   *
   * @throws IllegalArgumentException if {@code x <= 0}
   * @throws ArithmeticException if {@code mode} is {@link RoundingMode#UNNECESSARY} and {@code x}
   *     is not a power of ten
   */
  @GwtIncompatible // need BigIntegerMath to adequately test
  @SuppressWarnings("fallthrough")
