// Source-based slice around line 1025
// Method: <com.google.common.reflect.TypeToken: Type canonicalizeWildcardsInType(Type)>

   *   <li>{@code canonicalize(canonicalize(A)) == canonicalize(A)}.
   * </ol>
   */
  private static Type canonicalizeTypeArg(TypeVariable<?> declaration, Type typeArg) {
    return typeArg instanceof WildcardType
        ? canonicalizeWildcardType(declaration, ((WildcardType) typeArg))
        : canonicalizeWildcardsInType(typeArg);
  }

  private static Type canonicalizeWildcardsInType(Type type) {
    if (type instanceof ParameterizedType) {
      return canonicalizeWildcardsInParameterizedType((ParameterizedType) type);
    }
    if (type instanceof GenericArrayType) {
      return Types.newArrayType(
          canonicalizeWildcardsInType(((GenericArrayType) type).getGenericComponentType()));
    }
    return type;
  }

