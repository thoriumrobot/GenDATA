// Source-based slice around line 53
// Method: <com.google.common.primitives.UnsignedInts: int flip(int)>

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
   *
   * <p><b>Note:</b> this method is now unnecessary and should be treated as deprecated; use the
   * equivalent {@link Integer#compareUnsigned(int, int)} method instead.
   *
