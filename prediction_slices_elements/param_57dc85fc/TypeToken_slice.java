// Source-based slice around line 174
// Method: <com.google.common.reflect.TypeToken: TypeToken of(Type)>

    this.runtimeType = checkNotNull(type);
  }

  /** Returns an instance of type token that wraps {@code type}. */
  public static <T> TypeToken<T> of(Class<T> type) {
    return new SimpleTypeToken<>(type);
  }

  /** Returns an instance of type token that wraps {@code type}. */
  public static TypeToken<?> of(Type type) {
    return new SimpleTypeToken<>(type);
  }

  /**
   * Returns the raw type of {@code T}. Formally speaking, if {@code T} is returned by {@link
   * java.lang.reflect.Method#getGenericReturnType}, the raw type is what's returned by {@link
   * java.lang.reflect.Method#getReturnType} of the same method object. Specifically:
   *
   * <ul>
   *   <li>If {@code T} is a {@code Class} itself, {@code T} itself is returned.
