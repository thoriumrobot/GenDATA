// Source-based slice around line 411
// Method: <com.google.common.hash.HashCode: String toString()>

   *
   * <p>Note that if the output is considered to be a single hexadecimal number, whether this string
   * is big-endian or little-endian depends on the byte order of {@link #asBytes}. This may be
   * surprising for implementations of {@code HashCode} that represent the number in big-endian
   * since everything else in the hashing API uniformly treats multibyte values as little-endian.
   *
   * <p>To create a {@code HashCode} from its string representation, see {@link #fromString}.
   */
  @Override
  public final String toString() {
    byte[] bytes = getBytesInternal();
    StringBuilder sb = new StringBuilder(2 * bytes.length);
    for (byte b : bytes) {
      sb.append(hexDigits[(b >> 4) & 0xf]).append(hexDigits[b & 0xf]);
    }
    return sb.toString();
  }

  private static final char[] hexDigits = "0123456789abcdef".toCharArray();
}
