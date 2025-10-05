// Source-based slice around line 86
// Method: <com.google.common.hash.FarmHashFingerprint64: long hashLength16(long,long,long)>

    } else {
      return hashLength65Plus(bytes, offset, length);
    }
  }

  private static long shiftMix(long val) {
    return val ^ (val >>> 47);
  }

  private static long hashLength16(long u, long v, long mul) {
    long a = (u ^ v) * mul;
    a ^= a >>> 47;
    long b = (v ^ a) * mul;
    b ^= b >>> 47;
    b *= mul;
    return b;
  }

  /**
   * Computes intermediate hash of 32 bytes of byte array from the given offset. Results are
