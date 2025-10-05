// Source-based slice around line 577
// Method: <com.google.common.primitives.Doubles: List asList(double)>

   *
   * <p>The returned list is serializable.
   *
   * <p><b>Note:</b> when possible, you should represent your data as an {@link
   * ImmutableDoubleArray} instead, which has an {@link ImmutableDoubleArray#asList asList} view.
   *
   * @param backingArray the array to back the list
   * @return a list view of the array
   */
  public static List<Double> asList(double... backingArray) {
    if (backingArray.length == 0) {
      return Collections.emptyList();
    }
    return new DoubleArrayAsList(backingArray);
  }

  private static final class DoubleArrayAsList extends AbstractList<Double>
      implements RandomAccess, Serializable {
    final double[] array;
    final int start;
