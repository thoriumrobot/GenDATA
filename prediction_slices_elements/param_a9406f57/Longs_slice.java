// Source-based slice around line 711
// Method: <com.google.common.primitives.Longs: List asList(long)>

   *
   * <p>The returned list is serializable.
   *
   * <p><b>Note:</b> when possible, you should represent your data as an {@link ImmutableLongArray}
   * instead, which has an {@link ImmutableLongArray#asList asList} view.
   *
   * @param backingArray the array to back the list
   * @return a list view of the array
   */
  public static List<Long> asList(long... backingArray) {
    if (backingArray.length == 0) {
      return Collections.emptyList();
    }
    return new LongArrayAsList(backingArray);
  }

  private static final class LongArrayAsList extends AbstractList<Long>
      implements RandomAccess, Serializable {
    final long[] array;
    final int start;
