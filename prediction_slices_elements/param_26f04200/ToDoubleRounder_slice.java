// Source-based slice around line 42
// Method: <com.google.common.math.ToDoubleRounder: X minus(X,X)>

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
    checkNotNull(mode, "mode");
    double roundArbitrarily = roundToDoubleArbitrarily(x);
    if (Double.isInfinite(roundArbitrarily)) {
      switch (mode) {
        case DOWN:
        case HALF_EVEN:
