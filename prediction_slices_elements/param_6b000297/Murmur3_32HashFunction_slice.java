// Source-based slice around line 408
// Method: <com.google.common.hash.Murmur3_32HashFunction: long codePointToFourUtf8Bytes(int)>

    @Override
    public HashCode hash() {
      checkState(!isDone);
      isDone = true;
      h1 ^= mixK1((int) buffer);
      return fmix(h1, length);
    }
  }

  private static long codePointToFourUtf8Bytes(int codePoint) {
    // codePoint has at most 21 bits
    return ((0xFL << 4) | (codePoint >>> 18))
        | ((0x80L | (0x3F & (codePoint >>> 12))) << 8)
        | ((0x80L | (0x3F & (codePoint >>> 6))) << 16)
        | ((0x80L | (0x3F & codePoint)) << 24);
  }

  private static long charToThreeUtf8Bytes(char c) {
    return ((0x7L << 5) | (c >>> 12))
        | ((0x80 | (0x3F & (c >>> 6))) << 8)
