// Source-based slice around line 420
// Method: com.google.common.hash.HashCode.hexDigits

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
