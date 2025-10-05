// Source-based slice around line 1224
// Method: <com.google.common.reflect.TypeToken: TypeToken getSubtypeFromLowerBounds(Class,Type[])>

      if (bound.isSubtypeOf(supertype)) {
        @SuppressWarnings({"rawtypes", "unchecked"}) // guarded by the isSubtypeOf check.
        TypeToken<? super T> result = bound.getSupertype((Class) supertype);
        return result;
      }
    }
    throw new IllegalArgumentException(supertype + " isn't a super type of " + this);
  }

  private TypeToken<? extends T> getSubtypeFromLowerBounds(Class<?> subclass, Type[] lowerBounds) {
    if (lowerBounds.length > 0) {
      @SuppressWarnings("unchecked") // T's lower bound is <? extends T>
      TypeToken<? extends T> bound = (TypeToken<? extends T>) of(lowerBounds[0]);
      // Java supports only one lowerbound anyway.
      return bound.getSubtype(subclass);
    }
    throw new IllegalArgumentException(subclass + " isn't a subclass of " + this);
  }

  private TypeToken<? super T> getArraySupertype(Class<? super T> supertype) {
