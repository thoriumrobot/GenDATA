// Source-based slice around line 286
// Method: <com.google.common.primitives.UnsignedBytes: Comparator lexicographicalComparatorJavaImpl()>

   * Arrays::compareUnsigned}.
   *
   * @since 2.0
   */
  public static Comparator<byte[]> lexicographicalComparator() {
    return LexicographicalComparatorHolder.BEST_COMPARATOR;
  }

  @VisibleForTesting
  static Comparator<byte[]> lexicographicalComparatorJavaImpl() {
    return LexicographicalComparatorHolder.PureJavaComparator.INSTANCE;
  }

  /**
   * Provides a lexicographical comparator implementation; either a Java implementation or a faster
   * implementation based on {@link Unsafe}.
   *
   * <p>Uses reflection to gracefully fall back to the Java implementation if {@code Unsafe} isn't
   * available.
   */
