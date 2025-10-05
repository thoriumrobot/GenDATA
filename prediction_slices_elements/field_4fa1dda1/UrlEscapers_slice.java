// Source-based slice around line 147
// Method: com.google.common.net.UrlEscapers.URL_FRAGMENT_ESCAPER

   * </ul>
   *
   * <p><b>Note:</b> Unlike other escapers, URL escapers produce <a
   * href="https://url.spec.whatwg.org/#percent-encode">uppercase</a> hexadecimal sequences.
   */
  public static Escaper urlFragmentEscaper() {
    return URL_FRAGMENT_ESCAPER;
  }

  private static final Escaper URL_FRAGMENT_ESCAPER =
      new PercentEscaper(URL_PATH_OTHER_SAFE_CHARS_LACKING_PLUS + "+/?", false);
}
