// Source-based slice around line 171
// Method: <com.google.common.hash.FarmHashFingerprint64: long hashLength65Plus(byte[],int,int)>

    long g = (y + load64(bytes, offset + length - 32)) * mul;
    long h = (z + load64(bytes, offset + length - 24)) * mul;
    return hashLength16(
        rotateRight(e + f, 43) + rotateRight(g, 30) + h, e + rotateRight(f + a, 18) + g, mul);
  }

  /*
   * Compute an 8-byte hash of a byte array of length greater than 64 bytes.
   */
  private static long hashLength65Plus(byte[] bytes, int offset, int length) {
    int seed = 81;
    // For strings over 64 bytes we loop. Internal state consists of 56 bytes: v, w, x, y, and z.
    long x = seed;
    @SuppressWarnings("ConstantOverflow")
    long y = seed * K1 + 113;
    long z = shiftMix(y * K2 + 113) * K2;
    long[] v = new long[2];
    long[] w = new long[2];
    x = x * K2 + load64(bytes, offset);

