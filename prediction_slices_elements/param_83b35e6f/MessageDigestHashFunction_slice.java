// Source-based slice around line 63
// Method: <com.google.common.hash.MessageDigestHashFunction: boolean supportsClone(MessageDigest)>

    this.toString = checkNotNull(toString);
    this.prototype = getMessageDigest(algorithmName);
    int maxLength = prototype.getDigestLength();
    checkArgument(
        bytes >= 4 && bytes <= maxLength, "bytes (%s) must be >= 4 and < %s", bytes, maxLength);
    this.bytes = bytes;
    this.supportsClone = supportsClone(prototype);
  }

  private static boolean supportsClone(MessageDigest digest) {
    try {
      Object unused = digest.clone();
      return true;
    } catch (CloneNotSupportedException e) {
      return false;
    }
  }

  @Override
  public int bits() {
