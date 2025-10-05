// Source-based slice around line 491
// Method: <com.google.common.net.InternetDomainName: boolean isRegistrySuffix()>

   *
   * <p>For considerations on whether the public suffix or registry suffix designation is more
   * suitable for your application, see <a
   * href="https://github.com/google/guava/wiki/InternetDomainNameExplained">this article</a>.
   *
   * @return {@code true} if this domain name appears exactly on the public suffix list as part of
   *     the registry suffix section (labelled "ICANN").
   * @since 23.3
   */
  public boolean isRegistrySuffix() {
    return registrySuffixIndex() == 0;
  }

  /**
   * Indicates whether this domain name ends in a {@linkplain #isRegistrySuffix() registry suffix},
   * including if it is a registry suffix itself. For example, returns {@code true} for {@code
   * www.google.com}, {@code foo.co.uk} and {@code com}, but not for {@code invalid} or {@code
   * google.invalid}.
   *
   * <p>Note that this method is equivalent to {@link #hasPublicSuffix()} because all registry
