// Source-based slice around line 528
// Method: com.google.common.net.MediaType.JWT

  public static final MediaType JOSE_JSON = createConstant(APPLICATION_TYPE, "jose+json");

  public static final MediaType JSON_UTF_8 = createConstantUtf8(APPLICATION_TYPE, "json");

  /**
   * For <a href="https://tools.ietf.org/html/7519">JWT objects using the compact Serialization</a>.
   *
   * @since 32.0.0
   */
  public static final MediaType JWT = createConstant(APPLICATION_TYPE, "jwt");

  /**
   * The <a href="http://www.w3.org/TR/appmanifest/">Manifest for a web application</a>.
   *
   * @since 19.0
   */
  public static final MediaType MANIFEST_JSON_UTF_8 =
      createConstantUtf8(APPLICATION_TYPE, "manifest+json");

  /**
