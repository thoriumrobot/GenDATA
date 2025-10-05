// Source-based slice around line 91
// Method: <com.google.common.primitives.Primitives: Set allWrapperTypes()>

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
  }

  /**
   * Returns {@code true} if {@code type} is one of the nine primitive-wrapper types, such as {@link
   * Integer}.
   *
   * @see Class#isPrimitive
   */
  public static boolean isWrapperType(Class<?> type) {
