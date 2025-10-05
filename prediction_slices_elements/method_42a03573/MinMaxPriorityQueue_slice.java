// Source-based slice around line 408
// Method: <com.google.common.collect.MinMaxPriorityQueue: MoveDesc removeAt(int)>

   * <p>Occasionally, in order to maintain the heap invariant, it must swap a later element of the
   * list with one before {@code index}. Under these circumstances it returns a pair of elements as
   * a {@link MoveDesc}. The first one is the element that was previously at the end of the heap and
   * is now at some position before {@code index}. The second element is the one that was swapped
   * down to replace the element at {@code index}. This fact is used by iterator.remove so as to
   * visit elements during a traversal once and only once.
   */
  @VisibleForTesting
  @CanIgnoreReturnValue
  @Nullable MoveDesc<E> removeAt(int index) {
    checkPositionIndex(index, size);
    modCount++;
    size--;
    if (size == index) {
      queue[size] = null;
      return null;
    }
    E actualLastElement = elementData(size);
    int lastElementAt = heapForIndex(size).swapWithConceptuallyLastElement(actualLastElement);
    if (lastElementAt == index) {
