// Source-based slice around line 125
// Method: <com.google.common.graph.ElementOrder: ElementOrder insertion()>

   * </ul>
   *
   * @since 29.0
   */
  public static <S> ElementOrder<S> stable() {
    return new ElementOrder<>(Type.STABLE, null);
  }

  /** Returns an instance which specifies that insertion ordering is guaranteed. */
  public static <S> ElementOrder<S> insertion() {
    return new ElementOrder<>(Type.INSERTION, null);
  }

  /**
   * Returns an instance which specifies that the natural ordering of the elements is guaranteed.
   */
  public static <S extends Comparable<? super S>> ElementOrder<S> natural() {
    return new ElementOrder<>(Type.SORTED, Ordering.<S>natural());
  }

