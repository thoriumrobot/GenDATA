// Source-based slice around line 82
// Method: <com.google.common.hash.MessageDigestHashFunction: MessageDigest getMessageDigest(String)>

  public int bits() {
    return bytes * Byte.SIZE;
  }

  @Override
  public String toString() {
    return toString;
  }

  private static MessageDigest getMessageDigest(String algorithmName) {
    try {
      return MessageDigest.getInstance(algorithmName);
    } catch (NoSuchAlgorithmException e) {
      throw new AssertionError(e);
    }
  }

  @Override
  public Hasher newHasher() {
    if (supportsClone) {
