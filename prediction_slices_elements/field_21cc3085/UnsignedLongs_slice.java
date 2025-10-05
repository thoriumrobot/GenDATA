// Source-based slice around line 54
// Method: com.google.common.primitives.UnsignedLongs.MAX_VALUE

 * @author Louis Wasserman
 * @author Brian Milch
 * @author Colin Evans
 * @since 10.0
 */
@GwtCompatible
public final class UnsignedLongs {
  private UnsignedLongs() {}

  public static final long MAX_VALUE = -1L; // Equivalent to 2^64 - 1

  /**
   * A (self-inverse) bijection which converts the ordering on unsigned longs to the ordering on
   * longs, that is, {@code a <= b} as unsigned longs if and only if {@code flip(a) <= flip(b)} as
   * signed longs.
   */
  private static long flip(long a) {
    return a ^ Long.MIN_VALUE;
  }

