// Source-based slice around line 430
// Method: <com.google.common.reflect.TypeToken: TypeToken getSubtype(Class)>

        (TypeToken<? super T>) resolveSupertype(toGenericType(superclass).runtimeType);
    return supertype;
  }

  /**
   * Returns subtype of {@code this} with {@code subclass} as the raw class. For example, if this is
   * {@code Iterable<String>} and {@code subclass} is {@code List}, {@code List<String>} is
   * returned.
   */
  public final TypeToken<? extends T> getSubtype(Class<?> subclass) {
    checkArgument(
        !(runtimeType instanceof TypeVariable), "Cannot get subtype of type variable <%s>", this);
    if (runtimeType instanceof WildcardType) {
      return getSubtypeFromLowerBounds(subclass, ((WildcardType) runtimeType).getLowerBounds());
    }
    // unwrap array type if necessary
    if (isArray()) {
      return getArraySubtype(subclass);
    }
    // At this point, it's either a raw class or parameterized type.
