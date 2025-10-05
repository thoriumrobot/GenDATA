// Source-based slice around line 281
// Method: <com.google.common.collect.Range: Range all()>


  private static final Range<Comparable> ALL = new Range<>(Cut.belowAll(), Cut.aboveAll());

  /**
   * Returns a range that contains every value of type {@code C}.
   *
   * @since 14.0
   */
  @SuppressWarnings("unchecked")
  public static <C extends Comparable<?>> Range<C> all() {
    return (Range) ALL;
  }

  /**
   * Returns a range that {@linkplain Range#contains(Comparable) contains} only the given value. The
   * returned range is {@linkplain BoundType#CLOSED closed} on both ends.
   *
   * @since 14.0
   */
  public static <C extends Comparable<?>> Range<C> singleton(C value) {
