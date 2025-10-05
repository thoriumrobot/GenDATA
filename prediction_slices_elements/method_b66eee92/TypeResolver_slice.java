// Source-based slice around line 230
// Method: <com.google.common.reflect.TypeResolver: Type[] resolveTypesInPlace(Type[])>

    } else if (type instanceof WildcardType) {
      return resolveWildcardType((WildcardType) type);
    } else {
      // if Class<?>, no resolution needed, we are done.
      return type;
    }
  }

  @CanIgnoreReturnValue
  Type[] resolveTypesInPlace(Type[] types) {
    for (int i = 0; i < types.length; i++) {
      types[i] = resolveType(types[i]);
    }
    return types;
  }

  private Type[] resolveTypes(Type[] types) {
    Type[] result = new Type[types.length];
    for (int i = 0; i < types.length; i++) {
      result[i] = resolveType(types[i]);
