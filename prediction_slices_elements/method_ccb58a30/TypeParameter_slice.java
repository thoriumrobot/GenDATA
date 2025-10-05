// Source-based slice around line 62
// Method: <com.google.common.reflect.TypeParameter: boolean equals(Object)>

    this.typeVariable = (TypeVariable<?>) type;
  }

  @Override
  public final int hashCode() {
    return typeVariable.hashCode();
  }

  @Override
  public final boolean equals(@Nullable Object o) {
    if (o instanceof TypeParameter) {
      TypeParameter<?> that = (TypeParameter<?>) o;
      return typeVariable.equals(that.typeVariable);
    }
    return false;
  }

  @Override
  public String toString() {
    return typeVariable.toString();
