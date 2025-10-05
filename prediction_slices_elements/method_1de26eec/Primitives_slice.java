// Source-based slice around line 66
// Method: <com.google.common.primitives.Primitives: void add(Map,Map,Class,Class)>

    add(primToWrap, wrapToPrim, int.class, Integer.class);
    add(primToWrap, wrapToPrim, long.class, Long.class);
    add(primToWrap, wrapToPrim, short.class, Short.class);
    add(primToWrap, wrapToPrim, void.class, Void.class);

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
