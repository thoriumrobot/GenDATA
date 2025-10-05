// Source-based slice around line 154
// Method: <com.google.common.io.BaseEncoding: String encode(byte[],int,int)>

  /** Encodes the specified byte array, and returns the encoded {@code String}. */
  public String encode(byte[] bytes) {
    return encode(bytes, 0, bytes.length);
  }

  /**
   * Encodes the specified range of the specified byte array, and returns the encoded {@code
   * String}.
   */
  public final String encode(byte[] bytes, int off, int len) {
    checkPositionIndexes(off, off + len, bytes.length);
    StringBuilder result = new StringBuilder(maxEncodedSize(len));
    try {
      encodeTo(result, bytes, off, len);
    } catch (IOException impossible) {
      throw new AssertionError(impossible);
    }
    return result.toString();
  }

