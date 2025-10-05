// Source-based slice around line 121
// Method: <com.google.common.hash.LittleEndianByteArray: boolean usingFastPath()>

        | ((source[offset + 2] & 0xFF) << 16)
        | ((source[offset + 3] & 0xFF) << 24);
  }

  /**
   * Indicates that the load and store operations will be very efficient because of use of VarHandle
   * or Unsafe. May be useful for calling code to fall back on an alternative implementation that is
   * slower than those implementations but faster than the pure-Java mask-and-shift.
   */
  static boolean usingFastPath() {
    return byteArray.usesFastPath();
  }

  /**
   * Common interface for retrieving a 64-bit long from a little-endian byte array.
   *
   * <p>This abstraction allows us to use single-instruction load and put when available, or fall
   * back on the slower approach of using Longs.fromBytes(byte...).
   */
  private interface LittleEndianBytes {
