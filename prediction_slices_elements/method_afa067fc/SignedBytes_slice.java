// Source-based slice around line 170
// Method: <com.google.common.primitives.SignedBytes: Comparator lexicographicalComparator()>

   * example, {@code [] < [0x01] < [0x01, 0x80] < [0x01, 0x7F] < [0x02]}. Values are treated as
   * signed.
   *
   * <p>The returned comparator is inconsistent with {@link Object#equals(Object)} (since arrays
   * support only identity equality), but it is consistent with {@link
   * java.util.Arrays#equals(byte[], byte[])}.
   *
   * @since 2.0
   */
  public static Comparator<byte[]> lexicographicalComparator() {
    return LexicographicalComparator.INSTANCE;
  }

  private enum LexicographicalComparator implements Comparator<byte[]> {
    INSTANCE;

    @Override
    public int compare(byte[] left, byte[] right) {
      int minLength = Math.min(left.length, right.length);
      for (int i = 0; i < minLength; i++) {
