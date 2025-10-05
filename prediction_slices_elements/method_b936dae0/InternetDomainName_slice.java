// Source-based slice around line 195
// Method: <com.google.common.net.InternetDomainName: int registrySuffixIndex()>

    return publicSuffixIndexLocal;
  }

  /**
   * The index in the {@link #parts()} list at which the registry suffix begins. For example, for
   * the domain name {@code myblog.blogspot.co.uk}, the value would be 2 (the index of the {@code
   * co} part). The value is negative (specifically, {@link #NO_SUFFIX_FOUND}) if no registry suffix
   * was found.
   */
  private int registrySuffixIndex() {
    int registrySuffixIndexLocal = registrySuffixIndexCache;
    if (registrySuffixIndexLocal == SUFFIX_NOT_INITIALIZED) {
      registrySuffixIndexCache =
          registrySuffixIndexLocal = findSuffixOfType(Optional.of(PublicSuffixType.REGISTRY));
    }
    return registrySuffixIndexLocal;
  }

  /**
   * Returns the index of the leftmost part of the suffix, or -1 if not found. Note that the value
