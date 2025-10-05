// Source-based slice around line 260
// Method: <com.google.common.net.InternetDomainName: InternetDomainName from(String)>

   *       href="https://tools.ietf.org/html/rfc1123#section-2">RFC 1123</a>.
   * </ul>
   *
   * @param domain A domain name (not IP address)
   * @throws IllegalArgumentException if {@code domain} is not syntactically valid according to
   *     {@link #isValid}
   * @since 10.0 (previously named {@code fromLenient})
   */
  @CanIgnoreReturnValue // TODO(b/219820829): consider removing
  public static InternetDomainName from(String domain) {
    return new InternetDomainName(checkNotNull(domain));
  }

  /**
   * Validation method used by {@code from} to ensure that the domain name is syntactically valid
   * according to RFC 1035.
   *
   * @return Is the domain name syntactically valid?
   */
  private static boolean validateSyntax(List<String> parts) {
