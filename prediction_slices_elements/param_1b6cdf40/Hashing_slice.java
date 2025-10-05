// Source-based slice around line 391
// Method: <com.google.common.hash.Hashing: String hmacToString(String,Key)>

   * array and the SHA-512 algorithm.
   *
   * @param key the key material of the secret key
   * @since 20.0
   */
  public static HashFunction hmacSha512(byte[] key) {
    return hmacSha512(new SecretKeySpec(checkNotNull(key), "HmacSHA512"));
  }

  private static String hmacToString(String methodName, Key key) {
    return "Hashing."
        + methodName
        + "(Key[algorithm="
        + key.getAlgorithm()
        + ", format="
        + key.getFormat()
        + "])";
  }

  /**
