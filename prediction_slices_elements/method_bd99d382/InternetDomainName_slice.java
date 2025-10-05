// Source-based slice around line 382
// Method: <com.google.common.net.InternetDomainName: boolean isPublicSuffix()>

   * registry suffixes, since domain name registries collectively control all internet domain names.
   *
   * <p>For considerations on whether the public suffix or registry suffix designation is more
   * suitable for your application, see <a
   * href="https://github.com/google/guava/wiki/InternetDomainNameExplained">this article</a>.
   *
   * @return {@code true} if this domain name appears exactly on the public suffix list
   * @since 6.0
   */
  public boolean isPublicSuffix() {
    return publicSuffixIndex() == 0;
  }

  /**
   * Indicates whether this domain name ends in a {@linkplain #isPublicSuffix() public suffix},
   * including if it is a public suffix itself. For example, returns {@code true} for {@code
   * www.google.com}, {@code foo.co.uk} and {@code com}, but not for {@code invalid} or {@code
   * google.invalid}. This is the recommended method for determining whether a domain is potentially
   * an addressable host.
   *
