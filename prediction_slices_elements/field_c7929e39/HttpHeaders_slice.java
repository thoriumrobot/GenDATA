// Source-based slice around line 621
// Method: com.google.common.net.HttpHeaders.X_XSS_PROTECTION

   * <p>When the new X-Download-Options header is present with the value {@code noopen}, the user is
   * prevented from opening a file download directly; instead, they must first save the file
   * locally.
   *
   * @since 24.1
   */
  public static final String X_DOWNLOAD_OPTIONS = "X-Download-Options";

  /** The HTTP {@code X-XSS-Protection} header field name. */
  public static final String X_XSS_PROTECTION = "X-XSS-Protection";

  /**
   * The HTTP <a
   * href="https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/X-DNS-Prefetch-Control">{@code
   * X-DNS-Prefetch-Control}</a> header controls DNS prefetch behavior. Value can be "on" or "off".
   * By default, DNS prefetching is "on" for HTTP pages and "off" for HTTPS pages.
   */
  public static final String X_DNS_PREFETCH_CONTROL = "X-DNS-Prefetch-Control";

  /**
