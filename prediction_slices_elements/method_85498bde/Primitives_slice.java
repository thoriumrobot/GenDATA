// Source-based slice around line 101
// Method: <com.google.common.primitives.Primitives: boolean isWrapperType(Class)>

    return WRAPPER_TO_PRIMITIVE_TYPE.keySet();
  }

  /**
   * Returns {@code true} if {@code type} is one of the nine primitive-wrapper types, such as {@link
   * Integer}.
   *
   * @see Class#isPrimitive
   */
  public static boolean isWrapperType(Class<?> type) {
    return WRAPPER_TO_PRIMITIVE_TYPE.containsKey(checkNotNull(type));
  }

  /**
   * Returns the corresponding wrapper type of {@code type} if it is a primitive type; otherwise
   * returns {@code type} itself. Idempotent.
   *
   * <pre>
   *     wrap(int.class) == Integer.class
   *     wrap(Integer.class) == Integer.class
