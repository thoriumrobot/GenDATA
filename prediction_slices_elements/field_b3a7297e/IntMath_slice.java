// Source-based slice around line 618
// Method: com.google.common.math.IntMath.FLOOR_SQRT_MAX_INT

            if (-FLOOR_SQRT_MAX_INT > b | b > FLOOR_SQRT_MAX_INT) {
              return limit;
            }
            b *= b;
          }
      }
    }
  }

  @VisibleForTesting static final int FLOOR_SQRT_MAX_INT = 46340;

  /**
   * Returns {@code n!}, that is, the product of the first {@code n} positive integers, {@code 1} if
   * {@code n == 0}, or {@link Integer#MAX_VALUE} if the result does not fit in a {@code int}.
   *
   * @throws IllegalArgumentException if {@code n < 0}
   */
  public static int factorial(int n) {
    checkNonNegative("n", n);
    return (n < factorials.length) ? factorials[n] : Integer.MAX_VALUE;
