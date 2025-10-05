// Source-based slice around line 502
// Method: com.google.common.net.MediaType.JAVASCRIPT_UTF_8

   * @since 26.0
   */
  public static final MediaType HAL_JSON = createConstant(APPLICATION_TYPE, "hal+json");

  /**
   * <a href="http://www.rfc-editor.org/rfc/rfc4329.txt">RFC 4329</a> declares this to be the
   * correct media type for JavaScript, but {@link #TEXT_JAVASCRIPT_UTF_8 text/javascript} may be
   * necessary in certain situations for compatibility.
   */
  public static final MediaType JAVASCRIPT_UTF_8 =
      createConstantUtf8(APPLICATION_TYPE, "javascript");

  /**
   * For <a href="https://tools.ietf.org/html/rfc7515">JWS or JWE objects using the Compact
   * Serialization</a>.
   *
   * @since 27.1
   */
  public static final MediaType JOSE = createConstant(APPLICATION_TYPE, "jose");

