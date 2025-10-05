// Source-based slice around line 45
// Method: <com.google.common.math.ToDoubleRounder: double roundToDouble(X,RoundingMode)>

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
        case HALF_DOWN:
        case HALF_UP:
          return Double.MAX_VALUE * sign(x);
