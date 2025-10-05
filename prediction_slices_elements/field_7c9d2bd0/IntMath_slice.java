// Source-based slice around line 52
// Method: com.google.common.math.IntMath.MAX_SIGNED_POWER_OF_TWO

 * <p>Similar functionality for {@code long} and for {@link BigInteger} can be found in {@link
 * LongMath} and {@link BigIntegerMath} respectively. For other common operations on {@code int}
 * values, see {@link com.google.common.primitives.Ints}.
 *
 * @author Louis Wasserman
 * @since 11.0
 */
@GwtCompatible
public final class IntMath {
  @VisibleForTesting static final int MAX_SIGNED_POWER_OF_TWO = 1 << (Integer.SIZE - 2);

  /**
   * Returns the smallest power of two greater than or equal to {@code x}. This is equivalent to
   * {@code checkedPow(2, log2(x, CEILING))}.
   *
   * @throws IllegalArgumentException if {@code x <= 0}
   * @throws ArithmeticException of the next-higher power of two is not representable as an {@code
   *     int}, i.e. when {@code x > 2^30}
   * @since 20.0
   */
