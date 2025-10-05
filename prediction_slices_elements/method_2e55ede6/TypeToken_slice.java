// Source-based slice around line 534
// Method: <com.google.common.reflect.TypeToken: boolean isArray()>

    } else { // to instanceof TypeVariable
      return false;
    }
  }

  /**
   * Returns true if this type is known to be an array type, such as {@code int[]}, {@code T[]},
   * {@code <? extends Map<String, Integer>[]>} etc.
   */
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
