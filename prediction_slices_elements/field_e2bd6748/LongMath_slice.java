// Source-based slice around line 758
// Method: com.google.common.math.LongMath.FLOOR_SQRT_MAX_LONG

            if (-FLOOR_SQRT_MAX_LONG > b | b > FLOOR_SQRT_MAX_LONG) {
              return limit;
            }
            b *= b;
          }
      }
    }
  }

  @VisibleForTesting static final long FLOOR_SQRT_MAX_LONG = 3037000499L;

  /**
   * Returns {@code n!}, that is, the product of the first {@code n} positive integers, {@code 1} if
   * {@code n == 0}, or {@link Long#MAX_VALUE} if the result does not fit in a {@code long}.
   *
   * @throws IllegalArgumentException if {@code n < 0}
   */
  @GwtIncompatible // TODO
  public static long factorial(int n) {
    checkNonNegative("n", n);
