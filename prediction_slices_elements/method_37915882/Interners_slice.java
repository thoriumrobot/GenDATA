// Source-based slice around line 94
// Method: <com.google.common.collect.Interners: InternerBuilder newBuilder()>

    public <E> Interner<E> build() {
      if (!strong) {
        mapMaker.weakKeys();
      }
      return new InternerImpl<>(mapMaker);
    }
  }

  /** Returns a fresh {@link InternerBuilder} instance. */
  public static InternerBuilder newBuilder() {
    return new InternerBuilder();
  }

  /**
   * Returns a new thread-safe interner which retains a strong reference to each instance it has
   * interned, thus preventing these instances from being garbage-collected. If this retention is
   * acceptable, this implementation may perform better than {@link #newWeakInterner}.
   */
  public static <E> Interner<E> newStrongInterner() {
    return newBuilder().strong().build();
