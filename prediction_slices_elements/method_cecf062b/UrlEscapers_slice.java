// Source-based slice around line 75
// Method: <com.google.common.net.UrlEscapers: Escaper urlFormParameterEscaper()>

   * <p>This escaper is suitable for escaping parameter names and values even when <a
   * href="https://www.w3.org/TR/html401/appendix/notes.html#h-B.2.2">using the non-standard
   * semicolon</a>, rather than the ampersand, as a parameter delimiter. Nevertheless, we recommend
   * using the ampersand unless you must interoperate with systems that require semicolons.
   *
   * <p><b>Note:</b> Unlike other escapers, URL escapers produce <a
   * href="https://url.spec.whatwg.org/#percent-encode">uppercase</a> hexadecimal sequences.
   *
   */
  public static Escaper urlFormParameterEscaper() {
    return URL_FORM_PARAMETER_ESCAPER;
  }

  private static final Escaper URL_FORM_PARAMETER_ESCAPER =
      new PercentEscaper(URL_FORM_PARAMETER_OTHER_SAFE_CHARS, true);

  /**
   * Returns an {@link Escaper} instance that escapes strings so they can be safely included in <a
   * href="https://url.spec.whatwg.org/#syntax-url-path-segment">URL path segments</a>. The returned
   * escaper escapes all non-ASCII characters, even though <a
