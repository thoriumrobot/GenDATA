// Source-based slice around line 1062
// Method: <com.google.common.reflect.TypeToken: Bounds every(Type[])>

    Class<?> rawType = (Class<?>) type.getRawType();
    TypeVariable<?>[] typeVars = rawType.getTypeParameters();
    Type[] typeArgs = type.getActualTypeArguments();
    for (int i = 0; i < typeArgs.length; i++) {
      typeArgs[i] = canonicalizeTypeArg(typeVars[i], typeArgs[i]);
    }
    return Types.newParameterizedTypeWithOwner(type.getOwnerType(), rawType, typeArgs);
  }

  private static Bounds every(Type[] bounds) {
    // Every bound must match. On any false, result is false.
    return new Bounds(bounds, false);
  }

  private static Bounds any(Type[] bounds) {
    // Any bound matches. On any true, result is true.
    return new Bounds(bounds, true);
  }

  private static final class Bounds {
