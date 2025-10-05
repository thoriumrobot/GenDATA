// Source-based slice around line 79
// Method: com.google.common.net.UrlEscapers.URL_FORM_PARAMETER_ESCAPER

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
   * href="https://url.spec.whatwg.org/#url-code-points">many of these are accepted in modern
   * URLs</a>. (<a href="https://url.spec.whatwg.org/#path-state">If the escaper were to leave these
   * characters unescaped, they would be escaped by the consumer at parse time, anyway.</a>)
   * Additionally, the escaper escapes the slash character ("/"). While slashes are acceptable in
