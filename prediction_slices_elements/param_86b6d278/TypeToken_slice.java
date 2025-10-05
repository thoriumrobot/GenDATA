// Source-based slice around line 1052
// Method: <com.google.common.reflect.TypeToken: ParameterizedType canonicalizeWildcardsInParameterizedType(ParameterizedType)>

    for (Type bound : type.getUpperBounds()) {
      if (!any(declared).isSubtypeOf(bound)) {
        upperBounds.add(canonicalizeWildcardsInType(bound));
      }
    }
    return new Types.WildcardTypeImpl(type.getLowerBounds(), upperBounds.toArray(new Type[0]));
  }

  private static ParameterizedType canonicalizeWildcardsInParameterizedType(
      ParameterizedType type) {
    Class<?> rawType = (Class<?>) type.getRawType();
    TypeVariable<?>[] typeVars = rawType.getTypeParameters();
    Type[] typeArgs = type.getActualTypeArguments();
    for (int i = 0; i < typeArgs.length; i++) {
      typeArgs[i] = canonicalizeTypeArg(typeVars[i], typeArgs[i]);
    }
    return Types.newParameterizedTypeWithOwner(type.getOwnerType(), rawType, typeArgs);
  }

  private static Bounds every(Type[] bounds) {
