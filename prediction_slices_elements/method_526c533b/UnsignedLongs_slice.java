// Source-based slice around line 159
// Method: <com.google.common.primitives.UnsignedLongs: Comparator lexicographicalComparator()>

   * example, {@code [] < [1L] < [1L, 2L] < [2L] < [1L << 63]}.
   *
   * <p>The returned comparator is inconsistent with {@link Object#equals(Object)} (since arrays
   * support only identity equality), but it is consistent with {@link Arrays#equals(long[],
   * long[])}.
   *
   * <p><b>Java 9+ users:</b> Use {@link Arrays#compareUnsigned(long[], long[])
   * Arrays::compareUnsigned}.
   */
  public static Comparator<long[]> lexicographicalComparator() {
    return LexicographicalComparator.INSTANCE;
  }

  enum LexicographicalComparator implements Comparator<long[]> {
    INSTANCE;

    @Override
    public int compare(long[] left, long[] right) {
      int minLength = Math.min(left.length, right.length);
      for (int i = 0; i < minLength; i++) {
