// Source-based slice around line 449
// Method: <com.google.common.primitives.Shorts: Comparator lexicographicalComparator()>

   * common prefix, or when one array is a prefix of the other, treats the shorter array as the
   * lesser. For example, {@code [] < [(short) 1] < [(short) 1, (short) 2] < [(short) 2]}.
   *
   * <p>The returned comparator is inconsistent with {@link Object#equals(Object)} (since arrays
   * support only identity equality), but it is consistent with {@link Arrays#equals(short[],
   * short[])}.
   *
   * @since 2.0
   */
  public static Comparator<short[]> lexicographicalComparator() {
    return LexicographicalComparator.INSTANCE;
  }

  private enum LexicographicalComparator implements Comparator<short[]> {
    INSTANCE;

    @Override
    public int compare(short[] left, short[] right) {
      int minLength = Math.min(left.length, right.length);
      for (int i = 0; i < minLength; i++) {
