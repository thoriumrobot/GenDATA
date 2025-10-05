// Source-based slice around line 321
// Method: com.google.common.net.HttpHeaders.X_CONTENT_SECURITY_POLICY_REPORT_ONLY


  /**
   * The HTTP nonstandard {@code X-Content-Security-Policy-Report-Only} header field name. It was
   * introduced in <a href="https://www.w3.org/TR/2011/WD-CSP-20111129/">CSP v.1</a> and used by the
   * Firefox until version 23 and the Internet Explorer version 10. Please, use {@link
   * #CONTENT_SECURITY_POLICY_REPORT_ONLY} to pass the CSP.
   *
   * @since 20.0
   */
  public static final String X_CONTENT_SECURITY_POLICY_REPORT_ONLY =
      "X-Content-Security-Policy-Report-Only";

  /**
   * The HTTP nonstandard {@code X-WebKit-CSP} header field name. It was introduced in <a
   * href="https://www.w3.org/TR/2011/WD-CSP-20111129/">CSP v.1</a> and used by the Chrome until
   * version 25. Please, use {@link #CONTENT_SECURITY_POLICY} to pass the CSP.
   *
   * @since 20.0
   */
  public static final String X_WEBKIT_CSP = "X-WebKit-CSP";
