// Source-based slice around line 395
// Method: <com.google.common.reflect.TypeToken: TypeSet getTypes()>

   * types are parameterized with proper type arguments.
   *
   * <p>Subtypes are always listed before supertypes. But the reverse is not true. A type isn't
   * necessarily a subtype of all the types following. Order between types without subtype
   * relationship is arbitrary and not guaranteed.
   *
   * <p>If this type is a type variable or wildcard, upper bounds that are themselves type variables
   * aren't included (their super interfaces and superclasses are).
   */
  public final TypeSet getTypes() {
    return new TypeSet();
  }

  /**
   * Returns the generic form of {@code superclass}. For example, if this is {@code
   * ArrayList<String>}, {@code Iterable<String>} is returned given the input {@code
   * Iterable.class}.
   */
  public final TypeToken<? super T> getSupertype(Class<? super T> superclass) {
    checkArgument(
