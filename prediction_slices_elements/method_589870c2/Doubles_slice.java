// Source-based slice around line 394
// Method: <com.google.common.primitives.Doubles: Comparator lexicographicalComparator()>

   * common prefix, or when one array is a prefix of the other, treats the shorter array as the
   * lesser. For example, {@code [] < [1.0] < [1.0, 2.0] < [2.0]}.
   *
   * <p>The returned comparator is inconsistent with {@link Object#equals(Object)} (since arrays
   * support only identity equality), but it is consistent with {@link Arrays#equals(double[],
   * double[])}.
   *
   * @since 2.0
   */
  public static Comparator<double[]> lexicographicalComparator() {
    return LexicographicalComparator.INSTANCE;
  }

  private enum LexicographicalComparator implements Comparator<double[]> {
    INSTANCE;

    @Override
    public int compare(double[] left, double[] right) {
      int minLength = Math.min(left.length, right.length);
      for (int i = 0; i < minLength; i++) {
