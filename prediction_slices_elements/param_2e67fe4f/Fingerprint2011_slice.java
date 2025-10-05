// Source-based slice around line 76
// Method: <com.google.common.hash.Fingerprint2011: long shiftMix(long)>

      result = fullFingerprint(bytes, offset, length);
    }

    long u = length >= 8 ? load64(bytes, offset) : K0;
    long v = length >= 9 ? load64(bytes, offset + length - 8) : K0;
    result = hash128to64(result + v, u);
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
