// Source-based slice around line 355
// Method: <com.google.common.primitives.Booleans: boolean[] toArray(Collection)>

   * Calling this method is as thread-safe as calling that method.
   *
   * <p><b>Note:</b> consider representing the collection as a {@link java.util.BitSet} instead.
   *
   * @param collection a collection of {@code Boolean} objects
   * @return an array containing the same values as {@code collection}, in the same order, converted
   *     to primitives
   * @throws NullPointerException if {@code collection} or any of its elements is null
   */
  public static boolean[] toArray(Collection<Boolean> collection) {
    if (collection instanceof BooleanArrayAsList) {
      return ((BooleanArrayAsList) collection).toBooleanArray();
    }

    Object[] boxedArray = collection.toArray();
    int len = boxedArray.length;
    boolean[] array = new boolean[len];
    for (int i = 0; i < len; i++) {
      // checkNotNull for GWT (do not optimize)
      array[i] = (Boolean) checkNotNull(boxedArray[i]);
