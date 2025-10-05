// Source-based slice around line 543
// Method: <com.google.common.reflect.TypeToken: boolean isPrimitive()>

  public final boolean isArray() {
    return getComponentType() != null;
  }

  /**
   * Returns true if this type is one of the nine primitive types (including {@code void}).
   *
   * @since 15.0
   */
  public final boolean isPrimitive() {
    return (runtimeType instanceof Class) && ((Class<?>) runtimeType).isPrimitive();
  }

  /**
   * Returns the corresponding wrapper type if this is a primitive type; otherwise returns {@code
   * this} itself. Idempotent.
   *
   * @since 15.0
   */
  public final TypeToken<T> wrap() {
