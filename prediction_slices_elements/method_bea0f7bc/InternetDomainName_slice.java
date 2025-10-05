// Source-based slice around line 516
// Method: <com.google.common.net.InternetDomainName: InternetDomainName registrySuffix()>

    return registrySuffixIndex() != NO_SUFFIX_FOUND;
  }

  /**
   * Returns the {@linkplain #isRegistrySuffix() registry suffix} portion of the domain name, or
   * {@code null} if no registry suffix is present.
   *
   * @since 23.3
   */
  public @Nullable InternetDomainName registrySuffix() {
    return hasRegistrySuffix() ? ancestor(registrySuffixIndex()) : null;
  }

  /**
   * Indicates whether this domain name ends in a {@linkplain #isRegistrySuffix() registry suffix},
   * while not being a registry suffix itself. For example, returns {@code true} for {@code
   * www.google.com}, {@code foo.co.uk} and {@code blogspot.com}, but not for {@code com}, {@code
   * co.uk}, or {@code google.invalid}.
   *
   * @since 23.3
