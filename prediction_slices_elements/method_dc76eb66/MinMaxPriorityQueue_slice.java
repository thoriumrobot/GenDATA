// Source-based slice around line 900
// Method: <com.google.common.collect.MinMaxPriorityQueue: Iterator iterator()>

   * speaking, impossible to make any hard guarantees in the presence of unsynchronized concurrent
   * modification. Fail-fast iterators throw {@code ConcurrentModificationException} on a
   * best-effort basis. Therefore, it would be wrong to write a program that depended on this
   * exception for its correctness: <i>the fail-fast behavior of iterators should be used only to
   * detect bugs.</i>
   *
   * @return an iterator over the elements contained in this collection
   */
  @Override
  public Iterator<E> iterator() {
    return new QueueIterator();
  }

  @Override
  public void clear() {
    for (int i = 0; i < size; i++) {
      queue[i] = null;
    }
    size = 0;
  }
