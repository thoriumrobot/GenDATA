// Source-based slice around line 620
// Method: <com.google.common.primitives.Shorts: List asList(short)>

   * <p>The returned list maintains the values, but not the identities, of {@code Short} objects
   * written to or read from it. For example, whether {@code list.get(0) == list.get(0)} is true for
   * the returned list is unspecified.
   *
   * <p>The returned list is serializable.
   *
   * @param backingArray the array to back the list
   * @return a list view of the array
   */
  public static List<Short> asList(short... backingArray) {
    if (backingArray.length == 0) {
      return Collections.emptyList();
    }
    return new ShortArrayAsList(backingArray);
  }

  private static final class ShortArrayAsList extends AbstractList<Short>
      implements RandomAccess, Serializable {
    final short[] array;
    final int start;
