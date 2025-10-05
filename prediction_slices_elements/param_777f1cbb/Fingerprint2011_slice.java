// Source-based slice around line 115
// Method: <com.google.common.hash.Fingerprint2011: long fullFingerprint(byte[],int,int)>

    seedA += part3;
    seedB += rotateRight(seedA, 23);
    output[0] = seedA + part4;
    output[1] = seedB + c;
  }

  /*
   * Compute an 8-byte hash of a byte array of length greater than 64 bytes.
   */
  private static long fullFingerprint(byte[] bytes, int offset, int length) {
    // For lengths over 64 bytes we hash the end first, and then as we
    // loop we keep 56 bytes of state: v, w, x, y, and z.
    long x = load64(bytes, offset);
    long y = load64(bytes, offset + length - 16) ^ K1;
    long z = load64(bytes, offset + length - 56) ^ K0;
    long[] v = new long[2];
    long[] w = new long[2];
    weakHashLength32WithSeeds(bytes, offset + length - 64, length, y, v);
    weakHashLength32WithSeeds(bytes, offset + length - 32, length * K1, K0, w);
    z += shiftMix(v[1]) * K1;
