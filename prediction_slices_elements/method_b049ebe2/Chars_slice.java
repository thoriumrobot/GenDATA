// Source-based slice around line 569
// Method: <com.google.common.primitives.Chars: List asList(char)>

   * <p>The returned list maintains the values, but not the identities, of {@code Character} objects
   * written to or read from it. For example, whether {@code list.get(0) == list.get(0)} is true for
   * the returned list is unspecified.
   *
   * <p>The returned list is serializable.
   *
   * @param backingArray the array to back the list
   * @return a list view of the array
   */
  public static List<Character> asList(char... backingArray) {
    if (backingArray.length == 0) {
      return Collections.emptyList();
    }
    return new CharArrayAsList(backingArray);
  }

  private static final class CharArrayAsList extends AbstractList<Character>
      implements RandomAccess, Serializable {
    final char[] array;
    final int start;
