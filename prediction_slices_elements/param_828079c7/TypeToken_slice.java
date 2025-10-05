// Source-based slice around line 295
// Method: <com.google.common.reflect.TypeToken: TypeToken resolveSupertype(Type)>

   * }
   */
  public final TypeToken<?> resolveType(Type type) {
    checkNotNull(type);
    // Being conservative here because the user could use resolveType() to resolve a type in an
    // invariant context.
    return of(getInvariantTypeResolver().resolveType(type));
  }

  private TypeToken<?> resolveSupertype(Type type) {
    TypeToken<?> supertype = of(getCovariantTypeResolver().resolveType(type));
    // super types' type mapping is a subset of type mapping of this type.
    supertype.covariantTypeResolver = covariantTypeResolver;
    supertype.invariantTypeResolver = invariantTypeResolver;
    return supertype;
  }

  /**
   * Returns the generic superclass of this type or {@code null} if the type represents {@link
   * Object} or an interface. This method is similar but different from {@link
