// Source-based slice around line 180
// Method: <com.google.common.net.InternetDomainName: int publicSuffixIndex()>

    this.parts = parts;
  }

  /**
   * The index in the {@link #parts()} list at which the public suffix begins. For example, for the
   * domain name {@code myblog.blogspot.co.uk}, the value would be 1 (the index of the {@code
   * blogspot} part). The value is negative (specifically, {@link #NO_SUFFIX_FOUND}) if no public
   * suffix was found.
   */
  private int publicSuffixIndex() {
    int publicSuffixIndexLocal = publicSuffixIndexCache;
    if (publicSuffixIndexLocal == SUFFIX_NOT_INITIALIZED) {
      publicSuffixIndexCache =
          publicSuffixIndexLocal = findSuffixOfType(Optional.<PublicSuffixType>absent());
    }
    return publicSuffixIndexLocal;
  }

  /**
   * The index in the {@link #parts()} list at which the registry suffix begins. For example, for
