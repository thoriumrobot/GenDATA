// Source-based slice around line 598
// Method: <com.google.common.reflect.TypeToken: Invokable method(Method)>

    }
    return of(componentType);
  }

  /**
   * Returns the {@link Invokable} for {@code method}, which must be a member of {@code T}.
   *
   * @since 14.0
   */
  public final Invokable<T, Object> method(Method method) {
    checkArgument(
        this.someRawTypeIsSubclassOf(method.getDeclaringClass()),
        "%s not declared by %s",
        method,
        this);
    return new Invokable.MethodInvokable<T>(method) {
      @Override
      Type getGenericReturnType() {
        return getCovariantTypeResolver().resolveType(super.getGenericReturnType());
      }
