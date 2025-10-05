// Source-based slice around line 404
// Method: <com.google.common.reflect.TypeToken: TypeToken getSupertype(Class)>

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
        this.someRawTypeIsSubclassOf(superclass),
        "%s is not a super class of %s",
        superclass,
        this);
    if (runtimeType instanceof TypeVariable) {
      return getSupertypeFromUpperBounds(superclass, ((TypeVariable<?>) runtimeType).getBounds());
    }
    if (runtimeType instanceof WildcardType) {
      return getSupertypeFromUpperBounds(superclass, ((WildcardType) runtimeType).getUpperBounds());
