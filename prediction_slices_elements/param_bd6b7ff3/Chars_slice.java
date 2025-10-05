// Source-based slice around line 531
// Method: <com.google.common.primitives.Chars: void rotate(char[],int,int,int)>

   * Collections.rotate(Chars.asList(array).subList(fromIndex, toIndex), distance)}, but is
   * considerably faster and avoids allocations and garbage collection.
   *
   * <p>The provided "distance" may be negative, which will rotate left.
   *
   * @throws IndexOutOfBoundsException if {@code fromIndex < 0}, {@code toIndex > array.length}, or
   *     {@code toIndex > fromIndex}
   * @since 32.0.0
   */
  public static void rotate(char[] array, int distance, int fromIndex, int toIndex) {
    // See Ints.rotate for more details about possible algorithms here.
    checkNotNull(array);
    checkPositionIndexes(fromIndex, toIndex, array.length);
    if (array.length <= 1) {
      return;
    }

    int length = toIndex - fromIndex;
    // Obtain m = (-distance mod length), a non-negative value less than "length". This is how many
    // places left to rotate.
