// Source-based slice around line 515
// Method: <com.google.common.math.BigIntegerMath: boolean fitsInLong(BigInteger)>

      }
    }
    return accum
        .multiply(BigInteger.valueOf(numeratorAccum))
        .divide(BigInteger.valueOf(denominatorAccum));
  }

  // Returns true if BigInteger.valueOf(x.longValue()).equals(x).
  @GwtIncompatible // TODO
  static boolean fitsInLong(BigInteger x) {
    return x.bitLength() <= Long.SIZE - 1;
  }

  private BigIntegerMath() {}
}
