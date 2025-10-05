// Source-based slice around line 424
// Method: <com.google.common.net.InternetDomainName: boolean isUnderPublicSuffix()>

   * www.google.com}, {@code foo.co.uk} and {@code myblog.blogspot.com}, but not for {@code com},
   * {@code co.uk}, {@code google.invalid}, or {@code blogspot.com}.
   *
   * <p>This method can be used to determine whether it will probably be possible to set cookies on
   * the domain, though even that depends on individual browsers' implementations of cookie
   * controls. See <a href="http://www.ietf.org/rfc/rfc2109.txt">RFC 2109</a> for details.
   *
   * @since 6.0
   */
  public boolean isUnderPublicSuffix() {
    return publicSuffixIndex() > 0;
  }

  /**
   * Indicates whether this domain name is composed of exactly one subdomain component followed by a
   * {@linkplain #isPublicSuffix() public suffix}. For example, returns {@code true} for {@code
   * google.com} {@code foo.co.uk}, and {@code myblog.blogspot.com}, but not for {@code
   * www.google.com}, {@code co.uk}, or {@code blogspot.com}.
   *
   * <p>This method can be used to determine whether a domain is probably the highest level for
