// Source-based slice around line 393
// Method: com.google.common.net.MediaType.THREE_GPP2_VIDEO

  public static final MediaType THREE_GPP_VIDEO = createConstant(VIDEO_TYPE, "3gpp");

  /**
   * The 3G2 multimedia container format. For more information, see <a
   * href="http://www.3gpp2.org/Public_html/specs/C.S0050-B_v1.0_070521.pdf#page=16">3GPP2
   * C.S0050-B</a>.
   *
   * @since 20.0
   */
  public static final MediaType THREE_GPP2_VIDEO = createConstant(VIDEO_TYPE, "3gpp2");

  /* application types */
  /**
   * As described in <a href="http://www.ietf.org/rfc/rfc3023.txt">RFC 3023</a>, this constant
   * ({@code application/xml}) is used for XML documents that are "unreadable by casual users."
   * {@link #XML_UTF_8} is provided for documents that may be read by users.
   *
   * @since 14.0
   */
  public static final MediaType APPLICATION_XML_UTF_8 = createConstantUtf8(APPLICATION_TYPE, "xml");
