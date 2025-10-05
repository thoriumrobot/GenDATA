// Source-based slice around line 680
// Method: com.google.common.math.IntMath.biggestBinomials

          result *= n - i;
          result /= i + 1;
        }
        return (int) result;
    }
  }

  // binomial(biggestBinomials[k], k) fits in an int, but not binomial(biggestBinomials[k]+1,k).
  @VisibleForTesting
  static final int[] biggestBinomials = {
    Integer.MAX_VALUE,
    Integer.MAX_VALUE,
    65536,
    2345,
    477,
    193,
    110,
    75,
    58,
    49,
