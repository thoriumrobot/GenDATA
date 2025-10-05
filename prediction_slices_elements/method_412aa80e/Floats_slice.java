// Source-based slice around line 391
// Method: <com.google.common.primitives.Floats: Comparator lexicographicalComparator()>

   * common prefix, or when one array is a prefix of the other, treats the shorter array as the
   * lesser. For example, {@code [] < [1.0f] < [1.0f, 2.0f] < [2.0f]}.
   *
   * <p>The returned comparator is inconsistent with {@link Object#equals(Object)} (since arrays
   * support only identity equality), but it is consistent with {@link Arrays#equals(float[],
   * float[])}.
   *
   * @since 2.0
   */
  public static Comparator<float[]> lexicographicalComparator() {
    return LexicographicalComparator.INSTANCE;
  }

  private enum LexicographicalComparator implements Comparator<float[]> {
    INSTANCE;

    @Override
    public int compare(float[] left, float[] right) {
      int minLength = Math.min(left.length, right.length);
      for (int i = 0; i < minLength; i++) {
