// Source-based slice around line 315
// Method: <com.google.common.hash.Hashing: HashFunction hmacMd5(byte[])>

   * and the MD5 algorithm.
   *
   * <p>If you are designing a new system that needs HMAC, prefer {@link #hmacSha256} or other
   * future-proof algorithms <a
   * href="https://datatracker.ietf.org/doc/html/rfc6151#section-2.3">over {@code hmacMd5}</a>.
   *
   * @param key the key material of the secret key
   * @since 20.0
   */
  public static HashFunction hmacMd5(byte[] key) {
    return hmacMd5(new SecretKeySpec(checkNotNull(key), "HmacMD5"));
  }

  /**
   * Returns a hash function implementing the Message Authentication Code (MAC) algorithm, using the
   * SHA-1 (160 hash bits) hash function and the given secret key.
   *
   * @param key the secret key
   * @throws IllegalArgumentException if the given key is inappropriate for initializing this MAC
   * @since 20.0
