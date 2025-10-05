// Source-based slice around line 52
// Method: com.google.common.math.LongMath.MAX_SIGNED_POWER_OF_TWO

 * <p>Similar functionality for {@code int} and for {@link BigInteger} can be found in {@link
 * IntMath} and {@link BigIntegerMath} respectively. For other common operations on {@code long}
 * values, see {@link com.google.common.primitives.Longs}.
 *
 * @author Louis Wasserman
 * @since 11.0
 */
@GwtCompatible
public final class LongMath {
  @VisibleForTesting static final long MAX_SIGNED_POWER_OF_TWO = 1L << (Long.SIZE - 2);

  /**
   * Returns the smallest power of two greater than or equal to {@code x}. This is equivalent to
   * {@code checkedPow(2, log2(x, CEILING))}.
   *
   * @throws IllegalArgumentException if {@code x <= 0}
   * @throws ArithmeticException of the next-higher power of two is not representable as a {@code
   *     long}, i.e. when {@code x > 2^62}
   * @since 20.0
   */
