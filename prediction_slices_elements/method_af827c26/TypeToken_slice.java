// Source-based slice around line 333
// Method: <com.google.common.reflect.TypeToken: TypeToken boundAsSuperclass(Type)>

    Type superclass = getRawType().getGenericSuperclass();
    if (superclass == null) {
      return null;
    }
    @SuppressWarnings("unchecked") // super class of T
    TypeToken<? super T> superToken = (TypeToken<? super T>) resolveSupertype(superclass);
    return superToken;
  }

  private @Nullable TypeToken<? super T> boundAsSuperclass(Type bound) {
    TypeToken<?> token = of(bound);
    if (token.getRawType().isInterface()) {
      return null;
    }
    @SuppressWarnings("unchecked") // only upper bound of T is passed in.
    TypeToken<? super T> superclass = (TypeToken<? super T>) token;
    return superclass;
  }

  /**
