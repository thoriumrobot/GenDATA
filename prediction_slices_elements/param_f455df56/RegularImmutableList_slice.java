// Source-based slice around line 85
// Method: <com.google.common.collect.RegularImmutableList: UnmodifiableListIterator listIterator(int)>

  // The fake cast to E is safe because the creation methods only allow E's
  @Override
  @SuppressWarnings("unchecked")
  public E get(int index) {
    return (E) array[index];
  }

  @SuppressWarnings("unchecked")
  @Override
  public UnmodifiableListIterator<E> listIterator(int index) {
    // for performance
    // The fake cast to E is safe because the creation methods only allow E's
    return (UnmodifiableListIterator<E>) Iterators.forArrayWithPosition(array, index);
  }

  @Override
  public Spliterator<E> spliterator() {
    return Spliterators.spliterator(array, SPLITERATOR_CHARACTERISTICS);
  }

