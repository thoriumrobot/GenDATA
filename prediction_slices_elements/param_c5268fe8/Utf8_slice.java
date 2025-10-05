// Source-based slice around line 123
// Method: <com.google.common.base.Utf8: boolean isWellFormed(byte[],int,int)>

  /**
   * Returns whether the given byte array slice is a well-formed UTF-8 byte sequence, as defined by
   * {@link #isWellFormed(byte[])}. Note that this can be false even when {@code
   * isWellFormed(bytes)} is true.
   *
   * @param bytes the input buffer
   * @param off the offset in the buffer of the first byte to read
   * @param len the number of bytes to read from the buffer
   */
  public static boolean isWellFormed(byte[] bytes, int off, int len) {
    int end = off + len;
    checkPositionIndexes(off, end, bytes.length);
    // Look for the first non-ASCII character.
    for (int i = off; i < end; i++) {
      if (bytes[i] < 0) {
        return isWellFormedSlowPath(bytes, i, end);
      }
    }
    return true;
  }
