// Source-based slice around line 193
// Method: <com.google.common.primitives.UnsignedInts: Comparator lexicographicalComparator()>

   * prefix, or when one array is a prefix of the other, treats the shorter array as the lesser. For
   * example, {@code [] < [1] < [1, 2] < [2] < [1 << 31]}.
   *
   * <p>The returned comparator is inconsistent with {@link Object#equals(Object)} (since arrays
   * support only identity equality), but it is consistent with {@link Arrays#equals(int[], int[])}.
   *
   * <p><b>Java 9+ users:</b> Use {@link Arrays#compareUnsigned(int[], int[])
   * Arrays::compareUnsigned}.
   */
  public static Comparator<int[]> lexicographicalComparator() {
    return LexicographicalComparator.INSTANCE;
  }

  enum LexicographicalComparator implements Comparator<int[]> {
    INSTANCE;

    @Override
    // A call to bare "min" or "max" would resolve to our varargs method, not to any static import.
    @SuppressWarnings("StaticImportPreferred")
    public int compare(int[] left, int[] right) {
