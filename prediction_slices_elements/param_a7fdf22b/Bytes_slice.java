// Source-based slice around line 249
// Method: <com.google.common.primitives.Bytes: List asList(byte)>

   * <p>The returned list maintains the values, but not the identities, of {@code Byte} objects
   * written to or read from it. For example, whether {@code list.get(0) == list.get(0)} is true for
   * the returned list is unspecified.
   *
   * <p>The returned list is serializable.
   *
   * @param backingArray the array to back the list
   * @return a list view of the array
   */
  public static List<Byte> asList(byte... backingArray) {
    if (backingArray.length == 0) {
      return Collections.emptyList();
    }
    return new ByteArrayAsList(backingArray);
  }

  private static final class ByteArrayAsList extends AbstractList<Byte>
      implements RandomAccess, Serializable {
    final byte[] array;
    final int start;
