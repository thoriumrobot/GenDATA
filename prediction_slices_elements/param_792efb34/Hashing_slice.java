// Source-based slice around line 351
// Method: <com.google.common.hash.Hashing: HashFunction hmacSha256(Key)>


  /**
   * Returns a hash function implementing the Message Authentication Code (MAC) algorithm, using the
   * SHA-256 (256 hash bits) hash function and the given secret key.
   *
   * @param key the secret key
   * @throws IllegalArgumentException if the given key is inappropriate for initializing this MAC
   * @since 20.0
   */
  public static HashFunction hmacSha256(Key key) {
    return new MacHashFunction("HmacSHA256", key, hmacToString("hmacSha256", key));
  }

  /**
   * Returns a hash function implementing the Message Authentication Code (MAC) algorithm, using the
   * SHA-256 (256 hash bits) hash function and a {@link SecretKeySpec} created from the given byte
   * array and the SHA-256 algorithm.
   *
   * @param key the key material of the secret key
   * @since 20.0
