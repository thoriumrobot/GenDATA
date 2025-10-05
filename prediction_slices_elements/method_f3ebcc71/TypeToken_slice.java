// Source-based slice around line 315
// Method: <com.google.common.reflect.TypeToken: TypeToken getGenericSuperclass()>

   * Class#getGenericSuperclass}. For example, {@code new TypeToken<StringArrayList>()
   * {}.getGenericSuperclass()} will return {@code new TypeToken<ArrayList<String>>() {}}; while
   * {@code StringArrayList.class.getGenericSuperclass()} will return {@code ArrayList<E>}, where
   * {@code E} is the type variable declared by class {@code ArrayList}.
   *
   * <p>If this type is a type variable or wildcard, its first upper bound is examined and returned
   * if the bound is a class or extends from a class. This means that the returned type could be a
   * type variable too.
   */
  final @Nullable TypeToken<? super T> getGenericSuperclass() {
    if (runtimeType instanceof TypeVariable) {
      // First bound is always the super class, if one exists.
      return boundAsSuperclass(((TypeVariable<?>) runtimeType).getBounds()[0]);
    }
    if (runtimeType instanceof WildcardType) {
      // wildcard has one and only one upper bound.
      return boundAsSuperclass(((WildcardType) runtimeType).getUpperBounds()[0]);
    }
    Type superclass = getRawType().getGenericSuperclass();
    if (superclass == null) {
