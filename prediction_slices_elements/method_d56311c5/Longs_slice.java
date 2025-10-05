// Source-based slice around line 537
// Method: <com.google.common.primitives.Longs: Comparator lexicographicalComparator()>

   * prefix, or when one array is a prefix of the other, treats the shorter array as the lesser. For
   * example, {@code [] < [1L] < [1L, 2L] < [2L]}.
   *
   * <p>The returned comparator is inconsistent with {@link Object#equals(Object)} (since arrays
   * support only identity equality), but it is consistent with {@link Arrays#equals(long[],
   * long[])}.
   *
   * @since 2.0
   */
  public static Comparator<long[]> lexicographicalComparator() {
    return LexicographicalComparator.INSTANCE;
  }

  private enum LexicographicalComparator implements Comparator<long[]> {
    INSTANCE;

    @Override
    public int compare(long[] left, long[] right) {
      int minLength = Math.min(left.length, right.length);
      for (int i = 0; i < minLength; i++) {
