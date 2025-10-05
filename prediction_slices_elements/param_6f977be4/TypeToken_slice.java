// Source-based slice around line 1276
// Method: <com.google.common.reflect.TypeToken: Type resolveTypeArgsForSubclass(Class)>

        requireNonNull(getComponentType()).getSubtype(subclassComponentType);
    @SuppressWarnings("unchecked") // component type is subtype, so is array type.
    TypeToken<? extends T> result =
        (TypeToken<? extends T>)
            // If we are passed with int[].class, don't turn it to GenericArrayType
            of(newArrayClassOrGenericArrayType(componentSubtype.runtimeType));
    return result;
  }

  private Type resolveTypeArgsForSubclass(Class<?> subclass) {
    // If both runtimeType and subclass are not parameterized, return subclass
    // If runtimeType is not parameterized but subclass is, process subclass as a parameterized type
    // If runtimeType is a raw type (i.e. is a parameterized type specified as a Class<?>), we
    // return subclass as a raw type
    if (runtimeType instanceof Class
        && ((subclass.getTypeParameters().length == 0)
            || (getRawType().getTypeParameters().length != 0))) {
      // no resolution needed
      return subclass;
    }
