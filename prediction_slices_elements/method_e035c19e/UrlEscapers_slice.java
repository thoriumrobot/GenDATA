// Source-based slice around line 112
// Method: <com.google.common.net.UrlEscapers: Escaper urlPathSegmentEscaper()>

   *   <li>The space character " " is converted into %20.
   *   <li>All other characters are converted into one or more bytes using UTF-8 encoding and each
   *       byte is then represented by the 3-character string "%XY", where "XY" is the two-digit,
   *       uppercase, hexadecimal representation of the byte value.
   * </ul>
   *
   * <p><b>Note:</b> Unlike other escapers, URL escapers produce <a
   * href="https://url.spec.whatwg.org/#percent-encode">uppercase</a> hexadecimal sequences.
   */
  public static Escaper urlPathSegmentEscaper() {
    return URL_PATH_SEGMENT_ESCAPER;
  }

  private static final Escaper URL_PATH_SEGMENT_ESCAPER =
      new PercentEscaper(URL_PATH_OTHER_SAFE_CHARS_LACKING_PLUS + "+", false);

  /**
   * Returns an {@link Escaper} instance that escapes strings so they can be safely included in a <a
   * href="https://url.spec.whatwg.org/#concept-url-fragment">URL fragment</a>. The returned escaper
   * escapes all non-ASCII characters.
