// Source-based slice around line 192
// Method: com.google.common.net.MediaType.WML_UTF_8


  public static final MediaType VCARD_UTF_8 = createConstantUtf8(TEXT_TYPE, "vcard");

  /**
   * UTF-8 encoded <a href="https://en.wikipedia.org/wiki/Wireless_Markup_Language">Wireless Markup
   * Language</a>.
   *
   * @since 13.0
   */
  public static final MediaType WML_UTF_8 = createConstantUtf8(TEXT_TYPE, "vnd.wap.wml");

  /**
   * As described in <a href="http://www.ietf.org/rfc/rfc3023.txt">RFC 3023</a>, this constant
   * ({@code text/xml}) is used for XML documents that are "readable by casual users." {@link
   * #APPLICATION_XML_UTF_8} is provided for documents that are intended for applications.
   */
  public static final MediaType XML_UTF_8 = createConstantUtf8(TEXT_TYPE, "xml");

  /**
   * As described in <a href="https://w3c.github.io/webvtt/#iana-text-vtt">the VTT spec</a>, this is
