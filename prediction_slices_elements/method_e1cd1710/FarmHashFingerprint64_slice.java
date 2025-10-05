// Source-based slice around line 142
// Method: <com.google.common.hash.FarmHashFingerprint64: long hashLength17to32(byte[],int,int)>

      byte b = bytes[offset + (length >> 1)];
      byte c = bytes[offset + (length - 1)];
      int y = (a & 0xFF) + ((b & 0xFF) << 8);
      int z = length + ((c & 0xFF) << 2);
      return shiftMix(y * K2 ^ z * K0) * K2;
    }
    return K2;
  }

  private static long hashLength17to32(byte[] bytes, int offset, int length) {
    long mul = K2 + length * 2L;
    long a = load64(bytes, offset) * K1;
    long b = load64(bytes, offset + 8);
    long c = load64(bytes, offset + length - 8) * mul;
    long d = load64(bytes, offset + length - 16) * K2;
    return hashLength16(
        rotateRight(a + b, 43) + rotateRight(c, 30) + d, a + rotateRight(b + K2, 18) + c, mul);
  }

  private static long hashLength33To64(byte[] bytes, int offset, int length) {
