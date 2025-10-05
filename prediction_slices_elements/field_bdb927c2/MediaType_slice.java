// Source-based slice around line 511
// Method: com.google.common.net.MediaType.JOSE

  public static final MediaType JAVASCRIPT_UTF_8 =
      createConstantUtf8(APPLICATION_TYPE, "javascript");

  /**
   * For <a href="https://tools.ietf.org/html/rfc7515">JWS or JWE objects using the Compact
   * Serialization</a>.
   *
   * @since 27.1
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
