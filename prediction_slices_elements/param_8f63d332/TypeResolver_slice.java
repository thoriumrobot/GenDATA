// Source-based slice around line 268
// Method: <com.google.common.reflect.TypeResolver: T expectArgument(Class,Object)>

    Type resolvedOwner = (owner == null) ? null : resolveType(owner);
    Type resolvedRawType = resolveType(type.getRawType());

    Type[] args = type.getActualTypeArguments();
    Type[] resolvedArgs = resolveTypes(args);
    return Types.newParameterizedTypeWithOwner(
        resolvedOwner, (Class<?>) resolvedRawType, resolvedArgs);
  }

  private static <T> T expectArgument(Class<T> type, Object arg) {
    try {
      return type.cast(arg);
    } catch (ClassCastException e) {
      throw new IllegalArgumentException(arg + " is not a " + type.getSimpleName());
    }
  }

  /** A TypeTable maintains mapping from {@link TypeVariable} to types. */
  private static class TypeTable {
    private final ImmutableMap<TypeVariableKey, Type> map;
