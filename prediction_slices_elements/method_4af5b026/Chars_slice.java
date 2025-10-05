// Source-based slice around line 399
// Method: <com.google.common.primitives.Chars: Comparator lexicographicalComparator()>

   * that follow any common prefix, or when one array is a prefix of the other, treats the shorter
   * array as the lesser. For example, {@code [] < ['a'] < ['a', 'b'] < ['b']}.
   *
   * <p>The returned comparator is inconsistent with {@link Object#equals(Object)} (since arrays
   * support only identity equality), but it is consistent with {@link Arrays#equals(char[],
   * char[])}.
   *
   * @since 2.0
   */
  public static Comparator<char[]> lexicographicalComparator() {
    return LexicographicalComparator.INSTANCE;
  }

  private enum LexicographicalComparator implements Comparator<char[]> {
    INSTANCE;

    @Override
    public int compare(char[] left, char[] right) {
      int minLength = Math.min(left.length, right.length);
      for (int i = 0; i < minLength; i++) {
