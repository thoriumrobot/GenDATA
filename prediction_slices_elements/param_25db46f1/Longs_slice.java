// Source-based slice around line 321
// Method: <com.google.common.primitives.Longs: long fromBytes(byte,byte,byte,byte,byte,byte,byte,byte)>

  }

  /**
   * Returns the {@code long} value whose byte representation is the given 8 bytes, in big-endian
   * order; equivalent to {@code Longs.fromByteArray(new byte[] {b1, b2, b3, b4, b5, b6, b7, b8})}.
   *
   * @since 7.0
   */
  public static long fromBytes(
      byte b1, byte b2, byte b3, byte b4, byte b5, byte b6, byte b7, byte b8) {
    return (b1 & 0xFFL) << 56
        | (b2 & 0xFFL) << 48
        | (b3 & 0xFFL) << 40
        | (b4 & 0xFFL) << 32
        | (b5 & 0xFFL) << 24
        | (b6 & 0xFFL) << 16
        | (b7 & 0xFFL) << 8
        | (b8 & 0xFFL);
  }

