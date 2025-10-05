// Source-based slice around line 82
// Method: <com.google.common.primitives.Primitives: Set allPrimitiveTypes()>

  }

  /**
   * Returns an immutable set of all nine primitive types (including {@code void}). Note that a
   * simpler way to test whether a {@code Class} instance is a member of this set is to call {@link
   * Class#isPrimitive}.
   *
   * @since 3.0
   */
  public static Set<Class<?>> allPrimitiveTypes() {
    return PRIMITIVE_TO_WRAPPER_TYPE.keySet();
  }

  /**
   * Returns an immutable set of all nine primitive-wrapper types (including {@link Void}).
   *
   * @since 3.0
   */
  public static Set<Class<?>> allWrapperTypes() {
    return WRAPPER_TO_PRIMITIVE_TYPE.keySet();
