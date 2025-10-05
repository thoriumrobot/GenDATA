// Source-based slice around line 40
// Method: com.google.common.net.UrlEscapers.URL_PATH_OTHER_SAFE_CHARS_LACKING_PLUS

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
   * The caller is responsible for <a
   * href="https://html.spec.whatwg.org/multipage/form-control-infrastructure.html#multipart-form-data">replacing
