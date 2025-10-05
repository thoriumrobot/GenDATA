// Source-based slice around line 275
// Method: <com.google.common.reflect.TypeToken: TypeToken where(TypeParameter,Class)>

   *
   * @param <X> The parameter type
   * @param typeParam the parameter type variable
   * @param typeArg the actual type to substitute
   */
  /*
   * TODO(cpovirk): Is there any way for us to support TypeParameter instances for type parameters
   * that have nullable bounds? See discussion on the other overload of this method.
   */
  public final <X> TypeToken<T> where(TypeParameter<X> typeParam, Class<X> typeArg) {
    return where(typeParam, of(typeArg));
  }

  /**
   * Resolves the given {@code type} against the type context represented by this type. For example:
   *
   * {@snippet :
   * new TypeToken<List<String>>() {}.resolveType(
   *     List.class.getMethod("get", int.class).getGenericReturnType())
   * => String.class
