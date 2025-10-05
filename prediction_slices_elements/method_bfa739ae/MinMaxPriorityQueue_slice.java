// Source-based slice around line 266
// Method: <com.google.common.collect.MinMaxPriorityQueue: boolean add(E)>

  /**
   * Adds the given element to this queue. If this queue has a maximum size, after adding {@code
   * element} the queue will automatically evict its greatest element (according to its comparator),
   * which may be {@code element} itself.
   *
   * @return {@code true} always
   */
  @CanIgnoreReturnValue
  @Override
  public boolean add(E element) {
    offer(element);
    return true;
  }

  @CanIgnoreReturnValue
  @Override
  public boolean addAll(Collection<? extends E> newElements) {
    boolean modified = false;
    for (E element : newElements) {
      offer(element);
