// Source-based slice around line 49
// Method: com.google.common.primitives.UnsignedInts.INT_MASK

 * <p>See the Guava User Guide article on <a
 * href="https://github.com/google/guava/wiki/PrimitivesExplained#unsigned-support">unsigned
 * primitive utilities</a>.
 *
 * @author Louis Wasserman
 * @since 11.0
 */
@GwtCompatible
public final class UnsignedInts {
  static final long INT_MASK = 0xffffffffL;

  private UnsignedInts() {}

  static int flip(int value) {
    return value ^ Integer.MIN_VALUE;
  }

  /**
   * Compares the two specified {@code int} values, treating them as unsigned values between {@code
   * 0} and {@code 2^32 - 1} inclusive.
