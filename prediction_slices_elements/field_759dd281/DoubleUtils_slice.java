// Source-based slice around line 58
// Method: com.google.common.math.DoubleUtils.EXPONENT_BIAS

  // Double#doubleToRawLongBits(double)} spec.
  static final long EXPONENT_MASK = 0x7ff0000000000000L;

  // The mask for the sign, according to the {@link
  // Double#doubleToRawLongBits(double)} spec.
  static final long SIGN_MASK = 0x8000000000000000L;

  static final int SIGNIFICAND_BITS = 52;

  static final int EXPONENT_BIAS = 1023;

  /** The implicit 1 bit that is omitted in significands of normal doubles. */
  static final long IMPLICIT_BIT = SIGNIFICAND_MASK + 1;

  static long getSignificand(double d) {
    checkArgument(isFinite(d), "not a normal value");
    int exponent = getExponent(d);
    long bits = doubleToRawLongBits(d);
    bits &= SIGNIFICAND_MASK;
    return (exponent == MIN_EXPONENT - 1) ? bits << 1 : bits | IMPLICIT_BIT;
