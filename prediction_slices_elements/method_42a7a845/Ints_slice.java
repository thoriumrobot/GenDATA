// Source-based slice around line 457
// Method: <com.google.common.primitives.Ints: Comparator lexicographicalComparator()>

   * compares, using {@link #compare(int, int)}), the first pair of values that follow any common
   * prefix, or when one array is a prefix of the other, treats the shorter array as the lesser. For
   * example, {@code [] < [1] < [1, 2] < [2]}.
   *
   * <p>The returned comparator is inconsistent with {@link Object#equals(Object)} (since arrays
   * support only identity equality), but it is consistent with {@link Arrays#equals(int[], int[])}.
   *
   * @since 2.0
   */
  public static Comparator<int[]> lexicographicalComparator() {
    return LexicographicalComparator.INSTANCE;
  }

  private enum LexicographicalComparator implements Comparator<int[]> {
    INSTANCE;

    @Override
    // A call to bare "min" or "max" would resolve to our varargs method, not to any static import.
    @SuppressWarnings("StaticImportPreferred")
    public int compare(int[] left, int[] right) {
