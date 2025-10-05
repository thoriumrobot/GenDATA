// Source-based slice around line 133
// Method: <com.google.common.math.DoubleUtils: double ensureNonNegative(double)>

     * the exponent. This is exactly the behavior we get from just adding signifRounded to bits
     * directly. If the exponent is MAX_DOUBLE_EXPONENT, we round up (correctly) to
     * Double.POSITIVE_INFINITY.
     */
    bits |= x.signum() & SIGN_MASK;
    return longBitsToDouble(bits);
  }

  /** Returns its argument if it is non-negative, zero if it is negative. */
  static double ensureNonNegative(double value) {
    checkArgument(!isNaN(value));
    return max(value, 0.0);
  }

  @VisibleForTesting static final long ONE_BITS = 0x3ff0000000000000L;
}
