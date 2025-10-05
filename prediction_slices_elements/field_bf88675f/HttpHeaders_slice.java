// Source-based slice around line 300
// Method: com.google.common.net.HttpHeaders.CONTENT_SECURITY_POLICY_REPORT_ONLY

   */
  public static final String CONTENT_SECURITY_POLICY = "Content-Security-Policy";

  /**
   * The HTTP <a href="http://w3.org/TR/CSP/#content-security-policy-report-only-header-field">
   * {@code Content-Security-Policy-Report-Only}</a> header field name.
   *
   * @since 15.0
   */
  public static final String CONTENT_SECURITY_POLICY_REPORT_ONLY =
      "Content-Security-Policy-Report-Only";

  /**
   * The HTTP nonstandard {@code X-Content-Security-Policy} header field name. It was introduced in
   * <a href="https://www.w3.org/TR/2011/WD-CSP-20111129/">CSP v.1</a> and used by the Firefox until
   * version 23 and the Internet Explorer version 10. Please, use {@link #CONTENT_SECURITY_POLICY}
   * to pass the CSP.
   *
   * @since 20.0
   */
