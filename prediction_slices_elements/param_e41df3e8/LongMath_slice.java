// Source-based slice around line 182
// Method: <com.google.common.math.LongMath: int log10Floor(long)>

      case HALF_UP:
      case HALF_EVEN:
        // sqrt(10) is irrational, so log10(x)-logFloor is never exactly 0.5
        return logFloor + lessThanBranchFree(halfPowersOf10[logFloor], x);
    }
    throw new AssertionError();
  }

  @GwtIncompatible // TODO
  static int log10Floor(long x) {
    /*
     * Based on Hacker's Delight Fig. 11-5, the two-table-lookup, branch-free implementation.
     *
     * The key idea is that based on the number of leading zeros (equivalently, floor(log2(x))), we
     * can narrow the possible floor(log10(x)) values to two. For example, if floor(log2(x)) is 6,
     * then 64 <= x < 128, so floor(log10(x)) is either 1 or 2.
     */
    int y = maxLog10ForLeadingZeros[Long.numberOfLeadingZeros(x)];
    /*
     * y is the higher of the two possible values of floor(log10(x)). If x < 10^y, then we want the
