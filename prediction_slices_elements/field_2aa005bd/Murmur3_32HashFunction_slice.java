// Source-based slice around line 427
// Method: com.google.common.hash.Murmur3_32HashFunction.serialVersionUID

        | ((0x80 | (0x3F & (c >>> 6))) << 8)
        | ((0x80 | (0x3F & c)) << 16);
  }

  private static long charToTwoUtf8Bytes(char c) {
    // c has at most 11 bits
    return ((0x3L << 6) | (c >>> 6)) | ((0x80 | (0x3F & c)) << 8);
  }

  private static final long serialVersionUID = 0L;
}
