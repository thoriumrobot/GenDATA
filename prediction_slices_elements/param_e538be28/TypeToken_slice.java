// Source-based slice around line 1167
// Method: <com.google.common.reflect.TypeToken: TypeToken toGenericType(Class)>


  /**
   * Returns the type token representing the generic type declaration of {@code cls}. For example:
   * {@code TypeToken.getGenericType(Iterable.class)} returns {@code Iterable<T>}.
   *
   * <p>If {@code cls} isn't parameterized and isn't a generic array, the type token of the class is
   * returned.
   */
  @VisibleForTesting
  static <T> TypeToken<? extends T> toGenericType(Class<T> cls) {
    if (cls.isArray()) {
      Type arrayOfGenericType =
          Types.newArrayType(
              // If we are passed with int[].class, don't turn it to GenericArrayType
              toGenericType(cls.getComponentType()).runtimeType);
      @SuppressWarnings("unchecked") // array is covariant
      TypeToken<? extends T> result = (TypeToken<? extends T>) of(arrayOfGenericType);
      return result;
    }
    TypeVariable<Class<T>>[] typeParams = cls.getTypeParameters();
