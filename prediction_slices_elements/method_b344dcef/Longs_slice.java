// Source-based slice around line 679
// Method: <com.google.common.primitives.Longs: long[] toArray(Collection)>

   * <p>Elements are copied from the argument collection as if by {@code collection.toArray()}.
   * Calling this method is as thread-safe as calling that method.
   *
   * @param collection a collection of {@code Number} instances
   * @return an array containing the same values as {@code collection}, in the same order, converted
   *     to primitives
   * @throws NullPointerException if {@code collection} or any of its elements is null
   * @since 1.0 (parameter was {@code Collection<Long>} before 12.0)
   */
  public static long[] toArray(Collection<? extends Number> collection) {
    if (collection instanceof LongArrayAsList) {
      return ((LongArrayAsList) collection).toLongArray();
    }

    Object[] boxedArray = collection.toArray();
    int len = boxedArray.length;
    long[] array = new long[len];
    for (int i = 0; i < len; i++) {
      // checkNotNull for GWT (do not optimize)
      array[i] = ((Number) checkNotNull(boxedArray[i])).longValue();
