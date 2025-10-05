// Source-based slice around line 571
// Method: <com.google.common.primitives.Floats: List asList(float)>

   *
   * <p>The returned list may have unexpected behavior if it contains {@code NaN}, or if {@code NaN}
   * is used as a parameter to any of its methods.
   *
   * <p>The returned list is serializable.
   *
   * @param backingArray the array to back the list
   * @return a list view of the array
   */
  public static List<Float> asList(float... backingArray) {
    if (backingArray.length == 0) {
      return Collections.emptyList();
    }
    return new FloatArrayAsList(backingArray);
  }

  private static final class FloatArrayAsList extends AbstractList<Float>
      implements RandomAccess, Serializable {
    final float[] array;
    final int start;
