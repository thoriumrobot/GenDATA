// Source-based slice around line 553
// Method: <com.google.common.reflect.TypeToken: TypeToken wrap()>

    return (runtimeType instanceof Class) && ((Class<?>) runtimeType).isPrimitive();
  }

  /**
   * Returns the corresponding wrapper type if this is a primitive type; otherwise returns {@code
   * this} itself. Idempotent.
   *
   * @since 15.0
   */
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
