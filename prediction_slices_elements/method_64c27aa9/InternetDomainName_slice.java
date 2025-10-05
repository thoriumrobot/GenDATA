// Source-based slice around line 506
// Method: <com.google.common.net.InternetDomainName: boolean hasRegistrySuffix()>

   * including if it is a registry suffix itself. For example, returns {@code true} for {@code
   * www.google.com}, {@code foo.co.uk} and {@code com}, but not for {@code invalid} or {@code
   * google.invalid}.
   *
   * <p>Note that this method is equivalent to {@link #hasPublicSuffix()} because all registry
   * suffixes are public suffixes <i>and</i> all public suffixes have registry suffixes.
   *
   * @since 23.3
   */
  public boolean hasRegistrySuffix() {
    return registrySuffixIndex() != NO_SUFFIX_FOUND;
  }

  /**
   * Returns the {@linkplain #isRegistrySuffix() registry suffix} portion of the domain name, or
   * {@code null} if no registry suffix is present.
   *
   * @since 23.3
   */
  public @Nullable InternetDomainName registrySuffix() {
