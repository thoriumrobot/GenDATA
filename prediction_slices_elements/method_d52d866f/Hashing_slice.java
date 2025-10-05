// Source-based slice around line 268
// Method: <com.google.common.hash.Hashing: HashFunction sha384()>

    static final HashFunction SHA_256 =
        new MessageDigestHashFunction("SHA-256", "Hashing.sha256()");
  }

  /**
   * Returns a hash function implementing the SHA-384 algorithm (384 hash bits).
   *
   * @since 19.0
   */
  public static HashFunction sha384() {
    return Sha384Holder.SHA_384;
  }

  private static final class Sha384Holder {
    static final HashFunction SHA_384 =
        new MessageDigestHashFunction("SHA-384", "Hashing.sha384()");
  }

  /** Returns a hash function implementing the SHA-512 algorithm (512 hash bits). */
  public static HashFunction sha512() {
