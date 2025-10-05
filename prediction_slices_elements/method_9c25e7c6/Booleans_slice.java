// Source-based slice around line 383
// Method: <com.google.common.primitives.Booleans: List asList(boolean)>

   *
   * <p>There are at most two distinct objects in this list, {@code (Boolean) true} and {@code
   * (Boolean) false}. Java guarantees that those are always represented by the same objects.
   *
   * <p>The returned list is serializable.
   *
   * @param backingArray the array to back the list
   * @return a list view of the array
   */
  public static List<Boolean> asList(boolean... backingArray) {
    if (backingArray.length == 0) {
      return Collections.emptyList();
    }
    return new BooleanArrayAsList(backingArray);
  }

  private static final class BooleanArrayAsList extends AbstractList<Boolean>
      implements RandomAccess, Serializable {
    final boolean[] array;
    final int start;
