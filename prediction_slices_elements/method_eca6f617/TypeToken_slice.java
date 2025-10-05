// Source-based slice around line 1202
// Method: <com.google.common.reflect.TypeToken: TypeResolver getInvariantTypeResolver()>


  private TypeResolver getCovariantTypeResolver() {
    TypeResolver resolver = covariantTypeResolver;
    if (resolver == null) {
      resolver = (covariantTypeResolver = TypeResolver.covariantly(runtimeType));
    }
    return resolver;
  }

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
