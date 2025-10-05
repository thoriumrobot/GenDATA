// Source-based slice around line 914
// Method: <com.google.common.collect.MinMaxPriorityQueue: Object[] toArray()>

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
    arraycopy(queue, 0, copyTo, 0, size);
    return copyTo;
  }

  /**
   * Returns the comparator used to order the elements in this queue. Obeys the general contract of
   * {@link PriorityQueue#comparator}, but returns {@link Ordering#natural} instead of {@code null}
   * to indicate natural ordering.
   */
