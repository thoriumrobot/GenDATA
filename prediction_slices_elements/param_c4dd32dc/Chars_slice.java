// Source-based slice around line 436
// Method: <com.google.common.primitives.Chars: char[] toArray(Collection)>

   *
   * <p>Elements are copied from the argument collection as if by {@code collection.toArray()}.
   * Calling this method is as thread-safe as calling that method.
   *
   * @param collection a collection of {@code Character} objects
   * @return an array containing the same values as {@code collection}, in the same order, converted
   *     to primitives
   * @throws NullPointerException if {@code collection} or any of its elements is null
   */
  public static char[] toArray(Collection<Character> collection) {
    if (collection instanceof CharArrayAsList) {
      return ((CharArrayAsList) collection).toCharArray();
    }

    Object[] boxedArray = collection.toArray();
    int len = boxedArray.length;
    char[] array = new char[len];
    for (int i = 0; i < len; i++) {
      // checkNotNull for GWT (do not optimize)
      array[i] = (Character) checkNotNull(boxedArray[i]);
