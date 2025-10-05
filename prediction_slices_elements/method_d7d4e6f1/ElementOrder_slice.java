// Source-based slice around line 140
// Method: <com.google.common.graph.ElementOrder: ElementOrder sorted(Comparator)>

   */
  public static <S extends Comparable<? super S>> ElementOrder<S> natural() {
    return new ElementOrder<>(Type.SORTED, Ordering.<S>natural());
  }

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
