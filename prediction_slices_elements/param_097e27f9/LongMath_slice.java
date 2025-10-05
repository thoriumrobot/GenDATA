// Source-based slice around line 863
// Method: <com.google.common.math.LongMath: long multiplyFraction(long,long,long)>

              numeratorBits = nBits;
            }
          }
          return multiplyFraction(result, numerator, denominator);
        }
    }
  }

  /** Returns (x * numerator / denominator), which is assumed to come out to an integral value. */
  static long multiplyFraction(long x, long numerator, long denominator) {
    if (x == 1) {
      return numerator / denominator;
    }
    long commonDivisor = gcd(x, denominator);
    x /= commonDivisor;
    denominator /= commonDivisor;
    // We know gcd(x, denominator) = 1, and x * numerator / denominator is exact,
    // so denominator must be a divisor of numerator.
    return x * (numerator / denominator);
  }
