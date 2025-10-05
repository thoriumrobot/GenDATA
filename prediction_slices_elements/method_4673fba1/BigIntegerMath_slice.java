// Source-based slice around line 56
// Method: <com.google.common.math.BigIntegerMath: BigInteger ceilingPowerOfTwo(BigInteger)>

@GwtCompatible
public final class BigIntegerMath {
  /**
   * Returns the smallest power of two greater than or equal to {@code x}. This is equivalent to
   * {@code BigInteger.valueOf(2).pow(log2(x, CEILING))}.
   *
   * @throws IllegalArgumentException if {@code x <= 0}
   * @since 20.0
   */
  public static BigInteger ceilingPowerOfTwo(BigInteger x) {
    return BigInteger.ZERO.setBit(log2(x, CEILING));
  }

  /**
   * Returns the largest power of two less than or equal to {@code x}. This is equivalent to {@code
   * BigInteger.valueOf(2).pow(log2(x, FLOOR))}.
   *
   * @throws IllegalArgumentException if {@code x <= 0}
   * @since 20.0
   */
