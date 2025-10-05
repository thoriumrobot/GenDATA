// Source-based slice around line 36
// Method: <com.google.common.math.ToDoubleRounder: int sign(X)>

@GwtIncompatible
abstract class ToDoubleRounder<X extends Number & Comparable<X>> {
  /**
   * Returns x rounded to either the greatest double less than or equal to the precise value of x,
   * or the least double greater than or equal to the precise value of x.
   */
  abstract double roundToDoubleArbitrarily(X x);

  /** Returns the sign of x: either -1, 0, or 1. */
  abstract int sign(X x);

  /** Returns d's value as an X, rounded with the specified mode. */
  abstract X toX(double d, RoundingMode mode);

  /** Returns a - b, guaranteed that both arguments are nonnegative. */
  abstract X minus(X a, X b);

  /** Rounds {@code x} to a {@code double}. */
  final double roundToDouble(X x, RoundingMode mode) {
    checkNotNull(x, "x");
