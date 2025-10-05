// Source-based slice around line 91
// Method: <com.google.common.hash.MessageDigestHashFunction: Hasher newHasher()>

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
      try {
        return new MessageDigestHasher((MessageDigest) prototype.clone(), bytes);
      } catch (CloneNotSupportedException e) {
        // falls through
      }
    }
    return new MessageDigestHasher(getMessageDigest(prototype.getAlgorithm()), bytes);
  }

