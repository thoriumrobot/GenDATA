// Source-based slice around line 40
// Method: <com.google.common.math.DoubleUtils: double nextDown(double)>

/**
 * Utilities for {@code double} primitives.
 *
 * @author Louis Wasserman
 */
@GwtIncompatible
final class DoubleUtils {
  private DoubleUtils() {}

  static double nextDown(double d) {
    return -Math.nextUp(-d);
  }

  // The mask for the significand, according to the {@link
  // Double#doubleToRawLongBits(double)} spec.
  static final long SIGNIFICAND_MASK = 0x000fffffffffffffL;

  // The mask for the exponent, according to the {@link
  // Double#doubleToRawLongBits(double)} spec.
  static final long EXPONENT_MASK = 0x7ff0000000000000L;
