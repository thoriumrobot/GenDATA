// Source-based slice around line 1194
// Method: <com.google.common.reflect.TypeToken: TypeResolver getCovariantTypeResolver()>

      TypeToken<? extends T> type =
          (TypeToken<? extends T>)
              of(Types.newParameterizedTypeWithOwner(ownerType, cls, typeParams));
      return type;
    } else {
      return of(cls);
    }
  }

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
