// Source-based slice around line 299
// Method: <com.google.common.hash.Hashing: HashFunction hmacMd5(Key)>

   *
   * <p>If you are designing a new system that needs HMAC, prefer {@link #hmacSha256} or other
   * future-proof algorithms <a
   * href="https://datatracker.ietf.org/doc/html/rfc6151#section-2.3">over {@code hmacMd5}</a>.
   *
   * @param key the secret key
   * @throws IllegalArgumentException if the given key is inappropriate for initializing this MAC
   * @since 20.0
   */
  public static HashFunction hmacMd5(Key key) {
    return new MacHashFunction("HmacMD5", key, hmacToString("hmacMd5", key));
  }

  /**
   * Returns a hash function implementing the Message Authentication Code (MAC) algorithm, using the
   * MD5 (128 hash bits) hash function and a {@link SecretKeySpec} created from the given byte array
   * and the MD5 algorithm.
   *
   * <p>If you are designing a new system that needs HMAC, prefer {@link #hmacSha256} or other
   * future-proof algorithms <a
