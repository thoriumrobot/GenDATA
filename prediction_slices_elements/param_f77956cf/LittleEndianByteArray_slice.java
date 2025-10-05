// Source-based slice around line 94
// Method: <com.google.common.hash.LittleEndianByteArray: void store64(byte[],int,long)>

  }

  /**
   * Store 8 bytes into the provided array at the indicated offset, using the value provided.
   *
   * @param sink the output byte array
   * @param offset the offset into the array at which to start writing
   * @param value the value to write
   */
  static void store64(byte[] sink, int offset, long value) {
    // We don't want to assert in production code.
    assert offset >= 0 && offset + 8 <= sink.length;
    // Delegates to the fast (unsafe)version or the fallback.
    byteArray.putLongLittleEndian(sink, offset, value);
  }

  /**
   * Load 4 bytes from the provided array at the indicated offset.
   *
   * @param source the input bytes
