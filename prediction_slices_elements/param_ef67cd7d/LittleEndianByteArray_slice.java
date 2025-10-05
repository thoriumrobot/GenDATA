// Source-based slice around line 108
// Method: <com.google.common.hash.LittleEndianByteArray: int load32(byte[],int)>

  }

  /**
   * Load 4 bytes from the provided array at the indicated offset.
   *
   * @param source the input bytes
   * @param offset the offset into the array at which to start
   * @return the value found in the array in the form of a long
   */
  static int load32(byte[] source, int offset) {
    // TODO(user): Measure the benefit of delegating this to LittleEndianBytes also.
    return (source[offset] & 0xFF)
        | ((source[offset + 1] & 0xFF) << 8)
        | ((source[offset + 2] & 0xFF) << 16)
        | ((source[offset + 3] & 0xFF) << 24);
  }

  /**
   * Indicates that the load and store operations will be very efficient because of use of VarHandle
   * or Unsafe. May be useful for calling code to fall back on an alternative implementation that is
