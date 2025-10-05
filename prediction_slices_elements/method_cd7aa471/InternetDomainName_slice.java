// Source-based slice around line 398
// Method: <com.google.common.net.InternetDomainName: boolean hasPublicSuffix()>

   * www.google.com}, {@code foo.co.uk} and {@code com}, but not for {@code invalid} or {@code
   * google.invalid}. This is the recommended method for determining whether a domain is potentially
   * an addressable host.
   *
   * <p>Note that this method is equivalent to {@link #hasRegistrySuffix()} because all registry
   * suffixes are public suffixes <i>and</i> all public suffixes have registry suffixes.
   *
   * @since 6.0
   */
  public boolean hasPublicSuffix() {
    return publicSuffixIndex() != NO_SUFFIX_FOUND;
  }

  /**
   * Returns the {@linkplain #isPublicSuffix() public suffix} portion of the domain name, or {@code
   * null} if no public suffix is present.
   *
   * @since 6.0
   */
  public @Nullable InternetDomainName publicSuffix() {
