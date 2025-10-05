// Source-based slice around line 865
// Method: <com.google.common.reflect.TypeToken: TypeToken rejectTypeVariables()>

    // except TypeVariable.
    return of(new TypeResolver().resolveType(runtimeType));
  }

  /**
   * Ensures that this type token doesn't contain type variables, which can cause unchecked type
   * errors for callers like {@link TypeToInstanceMap}.
   */
  @CanIgnoreReturnValue
  final TypeToken<T> rejectTypeVariables() {
    new TypeVisitor() {
      @Override
      void visitTypeVariable(TypeVariable<?> type) {
        throw new IllegalArgumentException(
            runtimeType + "contains a type variable and is not safe for the operation");
      }

      @Override
      void visitWildcardType(WildcardType type) {
        visit(type.getLowerBounds());
