// Source-based slice around line 543
// Method: <com.google.common.net.InternetDomainName: boolean isTopDomainUnderRegistrySuffix()>

   * {@linkplain #isRegistrySuffix() registry suffix}. For example, returns {@code true} for {@code
   * google.com}, {@code foo.co.uk}, and {@code blogspot.com}, but not for {@code www.google.com},
   * {@code co.uk}, or {@code myblog.blogspot.com}.
   *
   * <p><b>Warning:</b> This method should not be used to determine the probable highest level
   * parent domain for which cookies may be set. Use {@link #topPrivateDomain()} for that purpose.
   *
   * @since 23.3
   */
  public boolean isTopDomainUnderRegistrySuffix() {
    return registrySuffixIndex() == 1;
  }

  /**
   * Returns the portion of this domain name that is one level beneath the {@linkplain
   * #isRegistrySuffix() registry suffix}. For example, for {@code x.adwords.google.co.uk} it
   * returns {@code google.co.uk}, since {@code co.uk} is a registry suffix. Similarly, for {@code
   * myblog.blogspot.com} it returns {@code blogspot.com}, since {@code com} is a registry suffix.
   *
   * <p>If {@link #isTopDomainUnderRegistrySuffix()} is true, the current domain name instance is
