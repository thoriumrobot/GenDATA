// Source-based slice around line 141
// Method: com.google.common.net.InternetDomainName.registrySuffixIndexCache

  /**
   * Cached value of #registrySuffixIndex(). Do not use directly.
   *
   * <p>Since this field isn't {@code volatile}, if an instance of this class is shared across
   * threads before it is initialized, then each thread is likely to compute their own copy of the
   * value.
   */
  @SuppressWarnings("Immutable")
  @LazyInit
  private int registrySuffixIndexCache = SUFFIX_NOT_INITIALIZED;

  /** Constructor used to implement {@link #from(String)}, and from subclasses. */
  InternetDomainName(String name) {
    // Normalize:
    // * ASCII characters to lowercase
    // * All dot-like characters to '.'
    // * Strip trailing '.'

    name = Ascii.toLowerCase(DOTS_MATCHER.replaceFrom(name, '.'));

