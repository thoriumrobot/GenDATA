// Source-based slice around line 100
// Method: <com.google.common.hash.FarmHashFingerprint64: void weakHashLength32WithSeeds(byte[],int,long,long,long[])>

    b *= mul;
    return b;
  }

  /**
   * Computes intermediate hash of 32 bytes of byte array from the given offset. Results are
   * returned in the output array because when we last measured, this was 12% faster than allocating
   * new arrays every time.
   */
  private static void weakHashLength32WithSeeds(
      byte[] bytes, int offset, long seedA, long seedB, long[] output) {
    long part1 = load64(bytes, offset);
    long part2 = load64(bytes, offset + 8);
    long part3 = load64(bytes, offset + 16);
    long part4 = load64(bytes, offset + 24);

    seedA += part1;
    seedB = rotateRight(seedB + seedA + part4, 21);
    long c = seedA;
    seedA += part2;
