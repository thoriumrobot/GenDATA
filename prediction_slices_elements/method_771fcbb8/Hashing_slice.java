// Source-based slice around line 363
// Method: <com.google.common.hash.Hashing: HashFunction hmacSha256(byte[])>


  /**
   * Returns a hash function implementing the Message Authentication Code (MAC) algorithm, using the
   * SHA-256 (256 hash bits) hash function and a {@link SecretKeySpec} created from the given byte
   * array and the SHA-256 algorithm.
   *
   * @param key the key material of the secret key
   * @since 20.0
   */
  public static HashFunction hmacSha256(byte[] key) {
    return hmacSha256(new SecretKeySpec(checkNotNull(key), "HmacSHA256"));
  }

  /**
   * Returns a hash function implementing the Message Authentication Code (MAC) algorithm, using the
   * SHA-512 (512 hash bits) hash function and the given secret key.
   *
   * @param key the secret key
   * @throws IllegalArgumentException if the given key is inappropriate for initializing this MAC
   * @since 20.0
