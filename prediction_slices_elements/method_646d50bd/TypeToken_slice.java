// Source-based slice around line 1259
// Method: <com.google.common.reflect.TypeToken: TypeToken getArraySubtype(Class)>

        componentType.getSupertype(requireNonNull(supertype.getComponentType()));
    @SuppressWarnings("unchecked") // component type is super type, so is array type.
    TypeToken<? super T> result =
        (TypeToken<? super T>)
            // If we are passed with int[].class, don't turn it to GenericArrayType
            of(newArrayClassOrGenericArrayType(componentSupertype.runtimeType));
    return result;
  }

  private TypeToken<? extends T> getArraySubtype(Class<?> subclass) {
    Class<?> subclassComponentType = subclass.getComponentType();
    if (subclassComponentType == null) {
      throw new IllegalArgumentException(subclass + " does not appear to be a subtype of " + this);
    }
    // array is covariant. component type is subtype, so is the array type.
    // requireNonNull is safe because we call getArraySubtype only when isArray().
    TypeToken<?> componentSubtype =
        requireNonNull(getComponentType()).getSubtype(subclassComponentType);
    @SuppressWarnings("unchecked") // component type is subtype, so is array type.
    TypeToken<? extends T> result =
