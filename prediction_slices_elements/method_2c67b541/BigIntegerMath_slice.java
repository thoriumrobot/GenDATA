// Source-based slice around line 254
// Method: <com.google.common.math.BigIntegerMath: BigInteger sqrtFloor(BigInteger)>

         * and halfSquare are integers, this is equivalent to testing whether or not x <=
         * halfSquare.
         */
        return (halfSquare.compareTo(x) >= 0) ? sqrtFloor : sqrtFloor.add(BigInteger.ONE);
    }
    throw new AssertionError();
  }

  @GwtIncompatible // TODO
  private static BigInteger sqrtFloor(BigInteger x) {
    /*
     * Adapted from Hacker's Delight, Figure 11-1.
     *
     * Using DoubleUtils.bigToDouble, getting a double approximation of x is extremely fast, and
     * then we can get a double approximation of the square root. Then, we iteratively improve this
     * guess with an application of Newton's method, which sets guess := (guess + (x / guess)) / 2.
     * This iteration has the following two properties:
     *
     * a) every iteration (except potentially the first) has guess >= floor(sqrt(x)). This is
     * because guess' is the arithmetic mean of guess and x / guess, sqrt(x) is the geometric mean,
