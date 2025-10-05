// Source-based slice around line 224
// Method: <com.google.common.hash.Hashing: HashFunction md5()>

   *     despite its deprecation. But if you can choose your hash function, avoid MD5, which is
   *     neither fast nor secure. As of January 2017, we suggest:
   *     <ul>
   *       <li>For security:
   *           {@link Hashing#sha256} or a higher-level API.
   *       <li>For speed: {@link Hashing#goodFastHash}, though see its docs for caveats.
   *     </ul>
   */
  @Deprecated
  public static HashFunction md5() {
    return Md5Holder.MD5;
  }

  private static final class Md5Holder {
    static final HashFunction MD5 = new MessageDigestHashFunction("MD5", "Hashing.md5()");
  }

  /**
   * Returns a hash function implementing the SHA-1 algorithm (160 hash bits).
   *
