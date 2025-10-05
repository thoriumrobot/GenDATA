// Source-based slice around line 73
// Method: <com.google.common.hash.LittleEndianByteArray: long load64Safely(byte[],int,int)>

   * Similar to load64, but allows offset + 8 > input.length, padding the result with zeroes. This
   * has to explicitly reverse the order of the bytes as it packs them into the result which makes
   * it slower than the native version.
   *
   * @param input the input bytes
   * @param offset the offset into the array at which to start reading
   * @param length the number of bytes from the input to read
   * @return a long of a concatenated 8 bytes
   */
  static long load64Safely(byte[] input, int offset, int length) {
    long result = 0;
    // Due to the way we shift, we can stop iterating once we've run out of data, the rest
    // of the result already being filled with zeros.

    // This loop is critical to performance, so please check HashBenchmark if altering it.
    int limit = min(length, 8);
    for (int i = 0; i < limit; i++) {
      // Shift value left while iterating logically through the array.
      result |= (input[offset + i] & 0xFFL) << (i * 8);
    }
