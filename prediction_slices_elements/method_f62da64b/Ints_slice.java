// Source-based slice around line 659
// Method: <com.google.common.primitives.Ints: List asList(int)>

   *
   * <p>The returned list is serializable.
   *
   * <p><b>Note:</b> when possible, you should represent your data as an {@link ImmutableIntArray}
   * instead, which has an {@link ImmutableIntArray#asList asList} view.
   *
   * @param backingArray the array to back the list
   * @return a list view of the array
   */
  public static List<Integer> asList(int... backingArray) {
    if (backingArray.length == 0) {
      return Collections.emptyList();
    }
    return new IntArrayAsList(backingArray);
  }

  private static final class IntArrayAsList extends AbstractList<Integer>
      implements RandomAccess, Serializable {
    final int[] array;
    final int start;
