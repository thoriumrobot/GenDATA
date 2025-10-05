// Source-based slice around line 86
// Method: <com.google.common.collect.RegularImmutableSet: Spliterator spliterator()>

  // We're careful to put only E instances into the array in the mainline.
  // (In the backport, we don't need this suppression, but we keep it to minimize diffs.)
  @SuppressWarnings("unchecked")
  @Override
  public UnmodifiableIterator<E> iterator() {
    return (UnmodifiableIterator<E>) Iterators.forArray(elements);
  }

  @Override
  public Spliterator<E> spliterator() {
    return Spliterators.spliterator(elements, SPLITERATOR_CHARACTERISTICS);
  }

  @Override
  Object[] internalArray() {
    return elements;
  }

  @Override
  int internalArrayStart() {
