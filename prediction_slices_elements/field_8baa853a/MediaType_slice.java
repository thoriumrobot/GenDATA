// Source-based slice around line 721
// Method: com.google.common.net.MediaType.TAR

   *
   * <p>For SOAP 1.1 messages, see {@code XML_UTF_8} per <a
   * href="http://www.w3.org/TR/2000/NOTE-SOAP-20000508/">W3C Note on Simple Object Access Protocol
   * (SOAP) 1.1</a>
   *
   * @since 20.0
   */
  public static final MediaType SOAP_XML_UTF_8 = createConstantUtf8(APPLICATION_TYPE, "soap+xml");

  public static final MediaType TAR = createConstant(APPLICATION_TYPE, "x-tar");

  /**
   * <a href="https://tools.ietf.org/html/rfc8081">RFC 8081</a> declares {@link #FONT_WOFF
   * font/woff} to be the correct media type for WOFF, but this may be necessary in certain
   * situations for compatibility.
   *
   * @since 17.0
   */
  public static final MediaType WOFF = createConstant(APPLICATION_TYPE, "font-woff");

