// Source-based slice around line 706
// Method: com.google.common.net.MediaType.SKETCHUP

  public static final MediaType SHOCKWAVE_FLASH =
      createConstant(APPLICATION_TYPE, "x-shockwave-flash");

  /**
   * {@code skp} files produced by the 3D Modeling software <a
   * href="https://www.sketchup.com/">SketchUp</a>
   *
   * @since 13.0
   */
  public static final MediaType SKETCHUP = createConstant(APPLICATION_TYPE, "vnd.sketchup.skp");

  /**
   * As described in <a href="http://www.ietf.org/rfc/rfc3902.txt">RFC 3902</a>, this constant
   * ({@code application/soap+xml}) is used to identify SOAP 1.2 message envelopes that have been
   * serialized with XML 1.0.
   *
   * <p>For SOAP 1.1 messages, see {@code XML_UTF_8} per <a
   * href="http://www.w3.org/TR/2000/NOTE-SOAP-20000508/">W3C Note on Simple Object Access Protocol
   * (SOAP) 1.1</a>
   *
