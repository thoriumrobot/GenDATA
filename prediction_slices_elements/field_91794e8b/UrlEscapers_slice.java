// Source-based slice around line 38
// Method: com.google.common.net.UrlEscapers.URL_FORM_PARAMETER_OTHER_SAFE_CHARS

 * @since 15.0
 */
@GwtCompatible
public final class UrlEscapers {
  private UrlEscapers() {}

  // For each xxxEscaper() method, please add links to external reference pages
  // that are considered authoritative for the behavior of that escaper.

  static final String URL_FORM_PARAMETER_OTHER_SAFE_CHARS = "-_.*";

  static final String URL_PATH_OTHER_SAFE_CHARS_LACKING_PLUS =
      "-._~" // Unreserved characters.
          + "!$'()*,;&=" // The subdelim characters (excluding '+').
          + "@:"; // The gendelim characters permitted in paths.

  /**
   * Returns an {@link Escaper} instance that escapes strings so they can be safely included in <a
   * href="https://url.spec.whatwg.org/#application-x-www-form-urlencoded-percent-encode-set">URL
   * form parameter names and values</a>. Escaping is performed with the UTF-8 character encoding.
