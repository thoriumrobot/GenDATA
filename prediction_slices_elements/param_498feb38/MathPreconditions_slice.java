// Source-based slice around line 38
// Method: <com.google.common.math.MathPreconditions: long checkPositive(String,long)>

  @CanIgnoreReturnValue
  static int checkPositive(String role, int x) {
    if (x <= 0) {
      throw new IllegalArgumentException(role + " (" + x + ") must be > 0");
    }
    return x;
  }

  @CanIgnoreReturnValue
  static long checkPositive(String role, long x) {
    if (x <= 0) {
      throw new IllegalArgumentException(role + " (" + x + ") must be > 0");
    }
    return x;
  }

  @CanIgnoreReturnValue
  static BigInteger checkPositive(String role, BigInteger x) {
    if (x.signum() <= 0) {
      throw new IllegalArgumentException(role + " (" + x + ") must be > 0");
