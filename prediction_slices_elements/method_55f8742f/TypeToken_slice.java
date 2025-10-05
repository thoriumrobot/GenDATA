// Source-based slice around line 1210
// Method: <com.google.common.reflect.TypeToken: TypeToken getSupertypeFromUpperBounds(Class,Type[])>


  private TypeResolver getInvariantTypeResolver() {
    TypeResolver resolver = invariantTypeResolver;
    if (resolver == null) {
      resolver = (invariantTypeResolver = TypeResolver.invariantly(runtimeType));
    }
    return resolver;
  }

  private TypeToken<? super T> getSupertypeFromUpperBounds(
      Class<? super T> supertype, Type[] upperBounds) {
    for (Type upperBound : upperBounds) {
      @SuppressWarnings("unchecked") // T's upperbound is <? super T>.
      TypeToken<? super T> bound = (TypeToken<? super T>) of(upperBound);
      if (bound.isSubtypeOf(supertype)) {
        @SuppressWarnings({"rawtypes", "unchecked"}) // guarded by the isSubtypeOf check.
        TypeToken<? super T> result = bound.getSupertype((Class) supertype);
        return result;
      }
    }
