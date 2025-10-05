// Source-based slice around line 355
// Method: <com.google.common.reflect.TypeToken: ImmutableList getGenericInterfaces()>

   * TypeToken<List<String>>() {}.getGenericInterfaces()} will return a list that contains {@code
   * new TypeToken<Iterable<String>>() {}}; while {@code List.class.getGenericInterfaces()} will
   * return an array that contains {@code Iterable<T>}, where the {@code T} is the type variable
   * declared by interface {@code Iterable}.
   *
   * <p>If this type is a type variable or wildcard, its upper bounds are examined and those that
   * are either an interface or upper-bounded only by interfaces are returned. This means that the
   * returned types could include type variables too.
   */
  final ImmutableList<TypeToken<? super T>> getGenericInterfaces() {
    if (runtimeType instanceof TypeVariable) {
      return boundsAsInterfaces(((TypeVariable<?>) runtimeType).getBounds());
    }
    if (runtimeType instanceof WildcardType) {
      return boundsAsInterfaces(((WildcardType) runtimeType).getUpperBounds());
    }
    ImmutableList.Builder<TypeToken<? super T>> builder = ImmutableList.builder();
    for (Type interfaceType : getRawType().getGenericInterfaces()) {
      @SuppressWarnings("unchecked") // interface of T
      TypeToken<? super T> resolvedInterface =
