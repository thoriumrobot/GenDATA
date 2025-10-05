// Source-based slice around line 57
// Method: <com.google.common.reflect.TypeParameter: int hashCode()>

  final TypeVariable<?> typeVariable;

  protected TypeParameter() {
    Type type = capture();
    checkArgument(type instanceof TypeVariable, "%s should be a type variable.", type);
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
