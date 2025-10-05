// Source-based slice around line 278
// Method: <com.google.common.hash.Hashing: HashFunction sha512()>

    return Sha384Holder.SHA_384;
  }

  private static final class Sha384Holder {
    static final HashFunction SHA_384 =
        new MessageDigestHashFunction("SHA-384", "Hashing.sha384()");
  }

  /** Returns a hash function implementing the SHA-512 algorithm (512 hash bits). */
  public static HashFunction sha512() {
    return Sha512Holder.SHA_512;
  }

  private static final class Sha512Holder {
    static final HashFunction SHA_512 =
        new MessageDigestHashFunction("SHA-512", "Hashing.sha512()");
  }

  /**
   * Returns a hash function implementing the Message Authentication Code (MAC) algorithm, using the
