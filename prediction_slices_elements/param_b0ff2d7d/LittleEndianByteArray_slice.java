// Source-based slice around line 56
// Method: <com.google.common.hash.LittleEndianByteArray: long load64(byte[],int)>


  /**
   * Load 8 bytes into long in a little endian manner, from the substring between position and
   * position + 8. The array must have at least 8 bytes from offset (inclusive).
   *
   * @param input the input bytes
   * @param offset the offset into the array at which to start
   * @return a long of a concatenated 8 bytes
   */
  static long load64(byte[] input, int offset) {
    // We don't want this in production code as this is the most critical part of the loop.
    assert input.length >= offset + 8;
    // Delegates to the fast (unsafe) version or the fallback.
    return byteArray.getLongLittleEndian(input, offset);
  }

  /**
   * Similar to load64, but allows offset + 8 > input.length, padding the result with zeroes. This
   * has to explicitly reverse the order of the bytes as it packs them into the result which makes
   * it slower than the native version.
