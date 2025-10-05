// Source-based slice around line 138
// Method: com.google.common.math.DoubleUtils.ONE_BITS

    return longBitsToDouble(bits);
  }

  /** Returns its argument if it is non-negative, zero if it is negative. */
  static double ensureNonNegative(double value) {
    checkArgument(!isNaN(value));
    return max(value, 0.0);
  }

  @VisibleForTesting static final long ONE_BITS = 0x3ff0000000000000L;
}
