// Source-based slice around line 444
// Method: com.google.common.net.MediaType.EPUB

  /**
   * As described in the <a href="http://idpf.org/epub">International Digital Publishing Forum</a>
   * EPUB is the distribution and interchange format standard for digital publications and
   * documents. This media type is defined in the <a
   * href="http://www.idpf.org/epub/30/spec/epub30-ocf.html">EPUB Open Container Format</a>
   * specification.
   *
   * @since 15.0
   */
  public static final MediaType EPUB = createConstant(APPLICATION_TYPE, "epub+zip");

  public static final MediaType FORM_DATA =
      createConstant(APPLICATION_TYPE, "x-www-form-urlencoded");

  /**
   * As described in <a href="https://www.rsa.com/rsalabs/node.asp?id=2138">PKCS #12: Personal
   * Information Exchange Syntax Standard</a>, PKCS #12 defines an archive file format for storing
   * many cryptography objects as a single file.
   *
   * @since 15.0
