// Source-based slice around line 440
// Method: <com.google.common.net.InternetDomainName: boolean isTopPrivateDomain()>

   * google.com} {@code foo.co.uk}, and {@code myblog.blogspot.com}, but not for {@code
   * www.google.com}, {@code co.uk}, or {@code blogspot.com}.
   *
   * <p>This method can be used to determine whether a domain is probably the highest level for
   * which cookies may be set, though even that depends on individual browsers' implementations of
   * cookie controls. See <a href="http://www.ietf.org/rfc/rfc2109.txt">RFC 2109</a> for details.
   *
   * @since 6.0
   */
  public boolean isTopPrivateDomain() {
    return publicSuffixIndex() == 1;
  }

  /**
   * Returns the portion of this domain name that is one level beneath the {@linkplain
   * #isPublicSuffix() public suffix}. For example, for {@code x.adwords.google.co.uk} it returns
   * {@code google.co.uk}, since {@code co.uk} is a public suffix. Similarly, for {@code
   * myblog.blogspot.com} it returns the same domain, {@code myblog.blogspot.com}, since {@code
   * blogspot.com} is a public suffix.
   *
