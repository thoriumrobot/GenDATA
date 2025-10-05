// Source-based slice around line 59
// Method: <com.google.common.collect.AbstractSequentialIterator: T computeNext(T)>

  protected AbstractSequentialIterator(@Nullable T firstOrNull) {
    this.nextOrNull = firstOrNull;
  }

  /**
   * Returns the element that follows {@code previous}, or returns {@code null} if no elements
   * remain. This method is invoked during each call to {@link #next()} in order to compute the
   * result of a <i>future</i> call to {@code next()}.
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
