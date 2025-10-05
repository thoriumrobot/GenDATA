// Source-based slice around line 169
// Method: <com.google.common.reflect.TypeToken: TypeToken of(Class)>

      this.runtimeType = TypeResolver.covariantly(declaringClass).resolveType(captured);
    }
  }

  private TypeToken(Type type) {
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
