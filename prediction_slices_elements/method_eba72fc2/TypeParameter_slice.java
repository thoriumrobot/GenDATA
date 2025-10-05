// Source-based slice around line 71
// Method: <com.google.common.reflect.TypeParameter: String toString()>

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
  }
}
