// Source-based slice around line 67
// Method: <com.google.common.collect.AbstractSequentialIterator: T next()>

   */
  protected abstract @Nullable T computeNext(T previous);

  @Override
  public final boolean hasNext() {
    return nextOrNull != null;
  }

  @Override
  public final T next() {
    if (nextOrNull == null) {
      throw new NoSuchElementException();
    }
    T oldNext = nextOrNull;
    nextOrNull = computeNext(oldNext);
    return oldNext;
  }
}
