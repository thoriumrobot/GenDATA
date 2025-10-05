// Source-based slice around line 82
// Method: <com.google.common.hash.Fingerprint2011: long hash128to64(long,long)>

    return result == 0 || result == 1 ? result + ~1 : result;
  }

  private static long shiftMix(long val) {
    return val ^ (val >>> 47);
  }

  /** Implementation of Hash128to64 from util/hash/hash128to64.h */
  @VisibleForTesting
  static long hash128to64(long high, long low) {
    long a = (low ^ high) * K3;
    a ^= a >>> 47;
    long b = (high ^ a) * K3;
    b ^= b >>> 47;
    b *= K3;
    return b;
  }

  /**
   * Computes intermediate hash of 32 bytes of byte array from the given offset. Results are
