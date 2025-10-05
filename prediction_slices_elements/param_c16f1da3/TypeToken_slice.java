// Source-based slice around line 984
// Method: <com.google.common.reflect.TypeToken: boolean is(Type,TypeVariable)>

   *
   * <p>It appears that properly handling recursive type bounds in the presence of implicit type
   * bounds is not easy. For now we punt, hoping that this defect should rarely cause issues in real
   * code.
   *
   * @param formalType is {@code Foo<formalType>} a supertype of {@code Foo<T>}?
   * @param declaration The type variable in the context of a parameterized type. Used to infer type
   *     bound when {@code formalType} is a wildcard with implicit upper bound.
   */
  private boolean is(Type formalType, TypeVariable<?> declaration) {
    if (runtimeType.equals(formalType)) {
      return true;
    }
    if (formalType instanceof WildcardType) {
      WildcardType your = canonicalizeWildcardType(declaration, (WildcardType) formalType);
      // if "formalType" is <? extends Foo>, "this" can be:
      // Foo, SubFoo, <? extends Foo>, <? extends SubFoo>, <T extends Foo> or
      // <T extends SubFoo>.
      // if "formalType" is <? super Foo>, "this" can be:
      // Foo, SuperFoo, <? super Foo> or <? super SuperFoo>.
