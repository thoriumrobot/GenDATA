// Source-based slice around line 146
// Method: <com.google.common.io.BaseEncoding: String encode(byte[])>

   * @since 15.0
   */
  public static final class DecodingException extends IOException {
    DecodingException(@Nullable String message) {
      super(message);
    }
  }

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
