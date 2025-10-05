// Source-based slice around line 254
// Method: <com.google.common.hash.Hashing: HashFunction sha256()>

  public static HashFunction sha1() {
    return Sha1Holder.SHA_1;
  }

  private static final class Sha1Holder {
    static final HashFunction SHA_1 = new MessageDigestHashFunction("SHA-1", "Hashing.sha1()");
  }

  /** Returns a hash function implementing the SHA-256 algorithm (256 hash bits). */
  public static HashFunction sha256() {
    return Sha256Holder.SHA_256;
  }

  private static final class Sha256Holder {
    static final HashFunction SHA_256 =
        new MessageDigestHashFunction("SHA-256", "Hashing.sha256()");
  }

  /**
   * Returns a hash function implementing the SHA-384 algorithm (384 hash bits).
