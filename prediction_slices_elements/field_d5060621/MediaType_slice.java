// Source-based slice around line 495
// Method: com.google.common.net.MediaType.HAL_JSON


  public static final MediaType GZIP = createConstant(APPLICATION_TYPE, "x-gzip");

  /**
   * <a href="https://tools.ietf.org/html/draft-kelly-json-hal-08#section-3">JSON Hypertext
   * Application Language (HAL) documents</a>.
   *
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
