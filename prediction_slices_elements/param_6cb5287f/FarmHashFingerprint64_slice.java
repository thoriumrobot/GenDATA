// Source-based slice around line 117
// Method: <com.google.common.hash.FarmHashFingerprint64: long hashLength0to16(byte[],int,int)>

    seedB = rotateRight(seedB + seedA + part4, 21);
    long c = seedA;
    seedA += part2;
    seedA += part3;
    seedB += rotateRight(seedA, 44);
    output[0] = seedA + part4;
    output[1] = seedB + c;
  }

  private static long hashLength0to16(byte[] bytes, int offset, int length) {
    if (length >= 8) {
      long mul = K2 + length * 2L;
      long a = load64(bytes, offset) + K2;
      long b = load64(bytes, offset + length - 8);
      long c = rotateRight(b, 37) * mul + a;
      long d = (rotateRight(a, 25) + b) * mul;
      return hashLength16(c, d, mul);
    }
    if (length >= 4) {
      long mul = K2 + length * 2;
