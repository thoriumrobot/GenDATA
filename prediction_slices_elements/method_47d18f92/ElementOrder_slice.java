// Source-based slice around line 145
// Method: <com.google.common.graph.ElementOrder: Type type()>

  /**
   * Returns an instance which specifies that the ordering of the elements is guaranteed to be
   * determined by {@code comparator}.
   */
  public static <S> ElementOrder<S> sorted(Comparator<S> comparator) {
    return new ElementOrder<>(Type.SORTED, checkNotNull(comparator));
  }

  /** Returns the type of ordering used. */
  public Type type() {
    return type;
  }

  /**
   * Returns the {@link Comparator} used.
   *
   * @throws UnsupportedOperationException if comparator is not defined
   */
  public Comparator<T> comparator() {
    if (comparator != null) {
