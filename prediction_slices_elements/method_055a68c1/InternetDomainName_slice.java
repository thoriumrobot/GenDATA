// Source-based slice around line 408
// Method: <com.google.common.net.InternetDomainName: InternetDomainName publicSuffix()>

    return publicSuffixIndex() != NO_SUFFIX_FOUND;
  }

  /**
   * Returns the {@linkplain #isPublicSuffix() public suffix} portion of the domain name, or {@code
   * null} if no public suffix is present.
   *
   * @since 6.0
   */
  public @Nullable InternetDomainName publicSuffix() {
    return hasPublicSuffix() ? ancestor(publicSuffixIndex()) : null;
  }

  /**
   * Indicates whether this domain name ends in a {@linkplain #isPublicSuffix() public suffix},
   * while not being a public suffix itself. For example, returns {@code true} for {@code
   * www.google.com}, {@code foo.co.uk} and {@code myblog.blogspot.com}, but not for {@code com},
   * {@code co.uk}, {@code google.invalid}, or {@code blogspot.com}.
   *
   * <p>This method can be used to determine whether it will probably be possible to set cookies on
