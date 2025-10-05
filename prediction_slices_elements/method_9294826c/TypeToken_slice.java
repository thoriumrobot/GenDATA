// Source-based slice around line 1019
// Method: <com.google.common.reflect.TypeToken: Type canonicalizeTypeArg(TypeVariable,Type)>

   *       Enum<? extends Enum<E>}.
   *   <li>{@code canonicalize(t)} produces a "literal" supertype of t. For example: {@code Enum<?
   *       extends Enum<?>>} canonicalizes to {@code Enum<?>}, which is a supertype (if we disregard
   *       the upper bound is implicitly an Enum too).
   *   <li>If {@code canonicalize(A) == canonicalize(B)}, then {@code Foo<A>.isSubtypeOf(Foo<B>)}
   *       and vice versa. i.e. {@code A.is(B)} and {@code B.is(A)}.
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
