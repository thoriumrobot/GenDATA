// Source-based slice around line 70
// Method: <com.google.common.primitives.Primitives: void add(Map,Map,Class,Class)>


    PRIMITIVE_TO_WRAPPER_TYPE = Collections.unmodifiableMap(primToWrap);
    WRAPPER_TO_PRIMITIVE_TYPE = Collections.unmodifiableMap(wrapToPrim);
  }

  private static void add(
      Map<Class<?>, Class<?>> forward,
      Map<Class<?>, Class<?>> backward,
      Class<?> key,
      Class<?> value) {
    forward.put(key, value);
    backward.put(value, key);
  }

  /**
   * Returns an immutable set of all nine primitive types (including {@code void}). Note that a
   * simpler way to test whether a {@code Class} instance is a member of this set is to call {@link
   * Class#isPrimitive}.
   *
   * @since 3.0
