// Source-based slice around line 273
// Method: <com.google.common.collect.MinMaxPriorityQueue: boolean addAll(Collection)>

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
      modified = true;
    }
    return modified;
  }

  /**
   * Adds the given element to this queue. If this queue has a maximum size, after adding {@code
