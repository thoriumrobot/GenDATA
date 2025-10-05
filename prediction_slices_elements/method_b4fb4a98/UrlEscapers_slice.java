// Source-based slice around line 143
// Method: <com.google.common.net.UrlEscapers: Escaper urlFragmentEscaper()>

   *   <li>Fragments allow unescaped "/" and "?", so they remain the same.
   *   <li>All other characters are converted into one or more bytes using UTF-8 encoding and each
   *       byte is then represented by the 3-character string "%XY", where "XY" is the two-digit,
   *       uppercase, hexadecimal representation of the byte value.
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
