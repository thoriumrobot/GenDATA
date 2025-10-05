// Source-based slice around line 844
// Method: <com.google.common.reflect.TypeToken: int hashCode()>

  public boolean equals(@Nullable Object o) {
    if (o instanceof TypeToken) {
      TypeToken<?> that = (TypeToken<?>) o;
      return runtimeType.equals(that.runtimeType);
    }
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
