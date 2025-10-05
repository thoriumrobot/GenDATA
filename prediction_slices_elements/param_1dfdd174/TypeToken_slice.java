// Source-based slice around line 1307
// Method: <com.google.common.reflect.TypeToken: Type newArrayClassOrGenericArrayType(Type)>

    return new TypeResolver()
        .where(supertypeWithArgsFromSubtype, runtimeType)
        .resolveType(genericSubtype.runtimeType);
  }

  /**
   * Creates an array class if {@code componentType} is a class, or else, a {@link
   * GenericArrayType}. This is what Java7 does for generic array type parameters.
   */
  private static Type newArrayClassOrGenericArrayType(Type componentType) {
    return Types.JavaVersion.JAVA7.newArrayType(componentType);
  }

  private static final class SimpleTypeToken<T> extends TypeToken<T> {

    SimpleTypeToken(Type type) {
      super(type);
    }

    private static final long serialVersionUID = 0;
