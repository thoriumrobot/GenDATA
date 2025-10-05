// Source-based slice around line 78
// Method: <com.google.common.hash.MessageDigestHashFunction: String toString()>

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
      return MessageDigest.getInstance(algorithmName);
    } catch (NoSuchAlgorithmException e) {
      throw new AssertionError(e);
    }
  }
