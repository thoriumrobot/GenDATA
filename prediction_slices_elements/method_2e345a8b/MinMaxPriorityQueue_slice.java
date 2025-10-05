// Source-based slice around line 905
// Method: <com.google.common.collect.MinMaxPriorityQueue: void clear()>

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

  @Override
  @J2ktIncompatible // Incompatible return type change. Use inherited (unoptimized) implementation
  public Object[] toArray() {
    Object[] copyTo = new Object[size];
