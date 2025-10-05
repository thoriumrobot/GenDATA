// Source-based slice around line 62
// Method: <com.google.common.math.MathPreconditions: long checkNonNegative(String,long)>

  @CanIgnoreReturnValue
  static int checkNonNegative(String role, int x) {
    if (x < 0) {
      throw new IllegalArgumentException(role + " (" + x + ") must be >= 0");
    }
    return x;
  }

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
