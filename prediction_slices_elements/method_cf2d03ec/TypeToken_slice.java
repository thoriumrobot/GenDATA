// Source-based slice around line 854
// Method: <com.google.common.reflect.TypeToken: Object writeReplace()>

    return runtimeType.hashCode();
  }

  @Override
  public String toString() {
    return Types.toString(runtimeType);
  }

  /** Implemented to support serialization of subclasses. */
  protected Object writeReplace() {
    // TypeResolver just transforms the type to our own impls that are Serializable
    // except TypeVariable.
    return of(new TypeResolver().resolveType(runtimeType));
  }

  /**
   * Ensures that this type token doesn't contain type variables, which can cause unchecked type
   * errors for callers like {@link TypeToInstanceMap}.
   */
  @CanIgnoreReturnValue
