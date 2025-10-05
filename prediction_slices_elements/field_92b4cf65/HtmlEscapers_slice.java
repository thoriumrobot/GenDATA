// Source-based slice around line 59
// Method: com.google.common.html.HtmlEscapers.HTML_ESCAPER

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
          .addEscape('\'', "&#39;")
          .addEscape('&', "&amp;")
          .addEscape('<', "&lt;")
          .addEscape('>', "&gt;")
          .build();

  private HtmlEscapers() {}
