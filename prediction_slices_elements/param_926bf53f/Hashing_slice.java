// Source-based slice around line 327
// Method: <com.google.common.hash.Hashing: HashFunction hmacSha1(Key)>


  /**
   * Returns a hash function implementing the Message Authentication Code (MAC) algorithm, using the
   * SHA-1 (160 hash bits) hash function and the given secret key.
   *
   * @param key the secret key
   * @throws IllegalArgumentException if the given key is inappropriate for initializing this MAC
   * @since 20.0
   */
  public static HashFunction hmacSha1(Key key) {
    return new MacHashFunction("HmacSHA1", key, hmacToString("hmacSha1", key));
  }

  /**
   * Returns a hash function implementing the Message Authentication Code (MAC) algorithm, using the
   * SHA-1 (160 hash bits) hash function and a {@link SecretKeySpec} created from the given byte
   * array and the SHA-1 algorithm.
   *
   * @param key the key material of the secret key
   * @since 20.0
