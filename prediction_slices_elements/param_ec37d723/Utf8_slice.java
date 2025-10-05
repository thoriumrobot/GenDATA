// Source-based slice around line 110
// Method: <com.google.common.base.Utf8: boolean isWellFormed(byte[])>

   * Returns {@code true} if {@code bytes} is a <i>well-formed</i> UTF-8 byte sequence according to
   * Unicode 6.0. Note that this is a stronger criterion than simply whether the bytes can be
   * decoded. For example, some versions of the JDK decoder will accept "non-shortest form" byte
   * sequences, but encoding never reproduces these. Such byte sequences are <i>not</i> considered
   * well-formed.
   *
   * <p>This method returns {@code true} if and only if {@code Arrays.equals(bytes, new
   * String(bytes, UTF_8).getBytes(UTF_8))} does, but is more efficient in both time and space.
   */
  public static boolean isWellFormed(byte[] bytes) {
    return isWellFormed(bytes, 0, bytes.length);
  }

  /**
   * Returns whether the given byte array slice is a well-formed UTF-8 byte sequence, as defined by
   * {@link #isWellFormed(byte[])}. Note that this can be false even when {@code
   * isWellFormed(bytes)} is true.
   *
   * @param bytes the input buffer
   * @param off the offset in the buffer of the first byte to read
