// Source-based slice around line 375
// Method: <com.google.common.hash.Hashing: HashFunction hmacSha512(Key)>


  /**
   * Returns a hash function implementing the Message Authentication Code (MAC) algorithm, using the
   * SHA-512 (512 hash bits) hash function and the given secret key.
   *
   * @param key the secret key
   * @throws IllegalArgumentException if the given key is inappropriate for initializing this MAC
   * @since 20.0
   */
  public static HashFunction hmacSha512(Key key) {
    return new MacHashFunction("HmacSHA512", key, hmacToString("hmacSha512", key));
  }

  /**
   * Returns a hash function implementing the Message Authentication Code (MAC) algorithm, using the
   * SHA-512 (512 hash bits) hash function and a {@link SecretKeySpec} created from the given byte
   * array and the SHA-512 algorithm.
   *
   * @param key the key material of the secret key
   * @since 20.0
