// Source-based slice around line 52
// Method: <com.google.common.html.HtmlEscapers: Escaper htmlEscaper()>

   * attribute values and in <em>most</em> elements' text contents, provided that the HTML
   * document's character encoding can encode any non-ASCII code points in the input (as UTF-8 and
   * other Unicode encodings can).
   *
   * <p><b>Note:</b> This escaper only performs minimal escaping to make content structurally
   * compatible with HTML. Specifically, it does not perform entity replacement (symbolic or
   * numeric), so it does not replace non-ASCII code points with character references. This escaper
   * escapes only the following five ASCII characters: {@code '"&<>}.
   */
  public static Escaper htmlEscaper() {
    return HTML_ESCAPER;
  }

  // For each xxxEscaper() method, please add links to external reference pages
  // that are considered authoritative for the behavior of that escaper.

  private static final Escaper HTML_ESCAPER =
      Escapers.builder()
          .addEscape('"', "&quot;")
          // Note: "&apos;" is not defined in HTML 4.01.
