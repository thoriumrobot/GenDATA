// Source-based slice around line 73
// Method: <com.google.common.hash.MessageDigestHashFunction: int bits()>

    try {
      Object unused = digest.clone();
      return true;
    } catch (CloneNotSupportedException e) {
      return false;
    }
  }

  @Override
  public int bits() {
    return bytes * Byte.SIZE;
  }

  @Override
  public String toString() {
    return toString;
  }

  private static MessageDigest getMessageDigest(String algorithmName) {
    try {
