// Source-based slice around line 1234
// Method: <com.google.common.reflect.TypeToken: TypeToken getArraySupertype(Class)>

    if (lowerBounds.length > 0) {
      @SuppressWarnings("unchecked") // T's lower bound is <? extends T>
      TypeToken<? extends T> bound = (TypeToken<? extends T>) of(lowerBounds[0]);
      // Java supports only one lowerbound anyway.
      return bound.getSubtype(subclass);
    }
    throw new IllegalArgumentException(subclass + " isn't a subclass of " + this);
  }

  private TypeToken<? super T> getArraySupertype(Class<? super T> supertype) {
    // with component type, we have lost generic type information
    // Use raw type so that compiler allows us to call getSupertype()
    @SuppressWarnings("rawtypes")
    TypeToken componentType = getComponentType();
    // TODO(cpovirk): checkArgument?
    if (componentType == null) {
      throw new IllegalArgumentException(supertype + " isn't a super type of " + this);
    }
    // array is covariant. component type is super type, so is the array type.
    @SuppressWarnings("unchecked") // going from raw type back to generics
