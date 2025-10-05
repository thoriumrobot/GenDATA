// Source-based slice around line 405
// Method: com.google.common.net.MediaType.ATOM_UTF_8

  /**
   * As described in <a href="http://www.ietf.org/rfc/rfc3023.txt">RFC 3023</a>, this constant
   * ({@code application/xml}) is used for XML documents that are "unreadable by casual users."
   * {@link #XML_UTF_8} is provided for documents that may be read by users.
   *
   * @since 14.0
   */
  public static final MediaType APPLICATION_XML_UTF_8 = createConstantUtf8(APPLICATION_TYPE, "xml");

  public static final MediaType ATOM_UTF_8 = createConstantUtf8(APPLICATION_TYPE, "atom+xml");
  public static final MediaType BZIP2 = createConstant(APPLICATION_TYPE, "x-bzip2");

  /**
   * Files in the <a href="https://www.dartlang.org/articles/embedding-in-html/">dart</a>
   * programming language.
   *
   * @since 19.0
   */
  public static final MediaType DART_UTF_8 = createConstantUtf8(APPLICATION_TYPE, "dart");

