// Source-based slice around line 637
// Method: <com.google.common.reflect.TypeToken: Invokable constructor(Constructor)>

      }
    };
  }

  /**
   * Returns the {@link Invokable} for {@code constructor}, which must be a member of {@code T}.
   *
   * @since 14.0
   */
  public final Invokable<T, T> constructor(Constructor<?> constructor) {
    checkArgument(
        constructor.getDeclaringClass() == getRawType(),
        "%s not declared by %s",
        constructor,
        getRawType());
    return new Invokable.ConstructorInvokable<T>(constructor) {
      @Override
      Type getGenericReturnType() {
        return getCovariantTypeResolver().resolveType(super.getGenericReturnType());
      }
