// Source-based slice around line 298
// Method: <com.google.common.math.BigIntegerMath: BigInteger sqrtApproxWithDoubles(BigInteger)>

    }
    do {
      sqrt0 = sqrt1;
      sqrt1 = sqrt0.add(x.divide(sqrt0)).shiftRight(1);
    } while (sqrt1.compareTo(sqrt0) < 0);
    return sqrt0;
  }

  @GwtIncompatible // TODO
  private static BigInteger sqrtApproxWithDoubles(BigInteger x) {
    return DoubleMath.roundToBigInteger(Math.sqrt(DoubleUtils.bigToDouble(x)), HALF_EVEN);
  }

  /**
   * Returns {@code x}, rounded to a {@code double} with the specified rounding mode. If {@code x}
   * is precisely representable as a {@code double}, its {@code double} value will be returned;
   * otherwise, the rounding will choose between the two nearest representable values with {@code
   * mode}.
   *
   * <p>For the case of {@link RoundingMode#HALF_DOWN}, {@code HALF_UP}, and {@code HALF_EVEN},
