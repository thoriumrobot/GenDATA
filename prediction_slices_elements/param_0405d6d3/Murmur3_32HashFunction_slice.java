// Source-based slice around line 416
// Method: <com.google.common.hash.Murmur3_32HashFunction: long charToThreeUtf8Bytes(char)>


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
        | ((0x80 | (0x3F & c)) << 16);
  }

  private static long charToTwoUtf8Bytes(char c) {
    // c has at most 11 bits
    return ((0x3L << 6) | (c >>> 6)) | ((0x80 | (0x3F & c)) << 8);
  }

