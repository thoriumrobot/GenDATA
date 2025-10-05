// Source-based slice around line 372
// Method: <com.google.common.reflect.TypeToken: ImmutableList boundsAsInterfaces(Type[])>

    for (Type interfaceType : getRawType().getGenericInterfaces()) {
      @SuppressWarnings("unchecked") // interface of T
      TypeToken<? super T> resolvedInterface =
          (TypeToken<? super T>) resolveSupertype(interfaceType);
      builder.add(resolvedInterface);
    }
    return builder.build();
  }

  private ImmutableList<TypeToken<? super T>> boundsAsInterfaces(Type[] bounds) {
    ImmutableList.Builder<TypeToken<? super T>> builder = ImmutableList.builder();
    for (Type bound : bounds) {
      @SuppressWarnings("unchecked") // upper bound of T
      TypeToken<? super T> boundType = (TypeToken<? super T>) of(bound);
      if (boundType.getRawType().isInterface()) {
        builder.add(boundType);
      }
    }
    return builder.build();
  }
