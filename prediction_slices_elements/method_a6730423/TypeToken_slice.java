// Source-based slice around line 562
// Method: <com.google.common.reflect.TypeToken: boolean isWrapper()>

  public final TypeToken<T> wrap() {
    if (isPrimitive()) {
      @SuppressWarnings("unchecked") // this is a primitive class
      Class<T> type = (Class<T>) runtimeType;
      return of(Primitives.wrap(type));
    }
    return this;
  }

  private boolean isWrapper() {
    return Primitives.allWrapperTypes().contains(runtimeType);
  }

  /**
   * Returns the corresponding primitive type if this is a wrapper type; otherwise returns {@code
   * this} itself. Idempotent.
   *
   * @since 15.0
   */
  public final TypeToken<T> unwrap() {
