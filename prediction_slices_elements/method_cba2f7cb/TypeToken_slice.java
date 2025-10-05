// Source-based slice around line 572
// Method: <com.google.common.reflect.TypeToken: TypeToken unwrap()>

    return Primitives.allWrapperTypes().contains(runtimeType);
  }

  /**
   * Returns the corresponding primitive type if this is a wrapper type; otherwise returns {@code
   * this} itself. Idempotent.
   *
   * @since 15.0
   */
  public final TypeToken<T> unwrap() {
    if (isWrapper()) {
      @SuppressWarnings("unchecked") // this is a wrapper class
      Class<T> type = (Class<T>) runtimeType;
      return of(Primitives.unwrap(type));
    }
    return this;
  }

  /**
   * Returns the array component type if this type represents an array ({@code int[]}, {@code T[]},
