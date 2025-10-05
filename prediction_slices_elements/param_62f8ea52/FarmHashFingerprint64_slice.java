// Source-based slice around line 82
// Method: <com.google.common.hash.FarmHashFingerprint64: long shiftMix(long)>

        return hashLength17to32(bytes, offset, length);
      }
    } else if (length <= 64) {
      return hashLength33To64(bytes, offset, length);
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
