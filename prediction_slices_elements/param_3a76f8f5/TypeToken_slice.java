// Source-based slice around line 288
// Method: <com.google.common.reflect.TypeToken: TypeToken resolveType(Type)>

  /**
   * Resolves the given {@code type} against the type context represented by this type. For example:
   *
   * {@snippet :
   * new TypeToken<List<String>>() {}.resolveType(
   *     List.class.getMethod("get", int.class).getGenericReturnType())
   * => String.class
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
