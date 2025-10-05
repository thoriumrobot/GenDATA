// Source-based slice around line 849
// Method: <com.google.common.reflect.TypeToken: String toString()>

    return false;
  }

  @Override
  public int hashCode() {
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

