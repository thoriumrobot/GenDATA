// Source-based slice around line 663
// Method: com.google.common.net.MediaType.OPENSEARCH_DESCRIPTION_UTF_8

      createConstant(APPLICATION_TYPE, "vnd.oasis.opendocument.text");

  /**
   * <a href="https://tools.ietf.org/id/draft-ellermann-opensearch-01.html">OpenSearch</a>
   * Description files are XML files that describe how a website can be used as a search engine by
   * consumers (e.g. web browsers).
   *
   * @since 28.2
   */
  public static final MediaType OPENSEARCH_DESCRIPTION_UTF_8 =
      createConstantUtf8(APPLICATION_TYPE, "opensearchdescription+xml");

  public static final MediaType PDF = createConstant(APPLICATION_TYPE, "pdf");
  public static final MediaType POSTSCRIPT = createConstant(APPLICATION_TYPE, "postscript");

  /**
   * <a href="http://tools.ietf.org/html/draft-rfernando-protocol-buffers-00">Protocol buffers</a>
   *
   * @since 15.0
   */
