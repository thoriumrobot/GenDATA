// Source-based slice around line 519
// Method: com.google.common.net.MediaType.JOSE_JSON

   */
  public static final MediaType JOSE = createConstant(APPLICATION_TYPE, "jose");

  /**
   * For <a href="https://tools.ietf.org/html/rfc7515">JWS or JWE objects using the JSON
   * Serialization</a>.
   *
   * @since 27.1
   */
  public static final MediaType JOSE_JSON = createConstant(APPLICATION_TYPE, "jose+json");

  public static final MediaType JSON_UTF_8 = createConstantUtf8(APPLICATION_TYPE, "json");

  /**
   * For <a href="https://tools.ietf.org/html/7519">JWT objects using the compact Serialization</a>.
   *
   * @since 32.0.0
   */
  public static final MediaType JWT = createConstant(APPLICATION_TYPE, "jwt");

