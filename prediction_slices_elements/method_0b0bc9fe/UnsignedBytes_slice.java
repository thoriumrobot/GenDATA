// Source-based slice around line 281
// Method: <com.google.common.primitives.UnsignedBytes: Comparator lexicographicalComparator()>

   * <p>The returned comparator is inconsistent with {@link Object#equals(Object)} (since arrays
   * support only identity equality), but it is consistent with {@link
   * java.util.Arrays#equals(byte[], byte[])}.
   *
   * <p><b>Java 9+ users:</b> Use {@link Arrays#compareUnsigned(byte[], byte[])
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
