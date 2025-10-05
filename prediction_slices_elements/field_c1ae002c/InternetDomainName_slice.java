// Source-based slice around line 130
// Method: com.google.common.net.InternetDomainName.publicSuffixIndexCache

  /**
   * Cached value of #publicSuffixIndex(). Do not use directly.
   *
   * <p>Since this field isn't {@code volatile}, if an instance of this class is shared across
   * threads before it is initialized, then each thread is likely to compute their own copy of the
   * value.
   */
  @SuppressWarnings("Immutable")
  @LazyInit
  private int publicSuffixIndexCache = SUFFIX_NOT_INITIALIZED;

  /**
   * Cached value of #registrySuffixIndex(). Do not use directly.
   *
   * <p>Since this field isn't {@code volatile}, if an instance of this class is shared across
   * threads before it is initialized, then each thread is likely to compute their own copy of the
   * value.
   */
  @SuppressWarnings("Immutable")
  @LazyInit
