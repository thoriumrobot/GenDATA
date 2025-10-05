// Source-based slice around line 70
// Method: <com.google.common.math.MathPreconditions: BigInteger checkNonNegative(String,BigInteger)>

  @CanIgnoreReturnValue
  static long checkNonNegative(String role, long x) {
    if (x < 0) {
      throw new IllegalArgumentException(role + " (" + x + ") must be >= 0");
    }
    return x;
  }

  @CanIgnoreReturnValue
  static BigInteger checkNonNegative(String role, BigInteger x) {
    if (x.signum() < 0) {
      throw new IllegalArgumentException(role + " (" + x + ") must be >= 0");
    }
    return x;
  }

  @CanIgnoreReturnValue
  static double checkNonNegative(String role, double x) {
    if (!(x >= 0)) { // not x < 0, to work with NaN.
      throw new IllegalArgumentException(role + " (" + x + ") must be >= 0");
