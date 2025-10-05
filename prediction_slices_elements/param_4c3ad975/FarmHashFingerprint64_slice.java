// Source-based slice around line 152
// Method: <com.google.common.hash.FarmHashFingerprint64: long hashLength33To64(byte[],int,int)>

    long mul = K2 + length * 2L;
    long a = load64(bytes, offset) * K1;
    long b = load64(bytes, offset + 8);
    long c = load64(bytes, offset + length - 8) * mul;
    long d = load64(bytes, offset + length - 16) * K2;
    return hashLength16(
        rotateRight(a + b, 43) + rotateRight(c, 30) + d, a + rotateRight(b + K2, 18) + c, mul);
  }

  private static long hashLength33To64(byte[] bytes, int offset, int length) {
    long mul = K2 + length * 2L;
    long a = load64(bytes, offset) * K2;
    long b = load64(bytes, offset + 8);
    long c = load64(bytes, offset + length - 8) * mul;
    long d = load64(bytes, offset + length - 16) * K2;
    long y = rotateRight(a + b, 43) + rotateRight(c, 30) + d;
    long z = hashLength16(y, a + rotateRight(b + K2, 18) + c, mul);
    long e = load64(bytes, offset + 16) * mul;
    long f = load64(bytes, offset + 24);
    long g = (y + load64(bytes, offset + length - 32)) * mul;
