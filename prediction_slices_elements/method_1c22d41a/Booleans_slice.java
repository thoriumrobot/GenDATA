// Source-based slice around line 314
// Method: <com.google.common.primitives.Booleans: Comparator lexicographicalComparator()>

   * common prefix, or when one array is a prefix of the other, treats the shorter array as the
   * lesser. For example, {@code [] < [false] < [false, true] < [true]}.
   *
   * <p>The returned comparator is inconsistent with {@link Object#equals(Object)} (since arrays
   * support only identity equality), but it is consistent with {@link Arrays#equals(boolean[],
   * boolean[])}.
   *
   * @since 2.0
   */
  public static Comparator<boolean[]> lexicographicalComparator() {
    return LexicographicalComparator.INSTANCE;
  }

  private enum LexicographicalComparator implements Comparator<boolean[]> {
    INSTANCE;

    @Override
    public int compare(boolean[] left, boolean[] right) {
      // do not static import Math.min due to https://bugs.openjdk.org/browse/JDK-8357219
      @SuppressWarnings("StaticImportPreferred")
