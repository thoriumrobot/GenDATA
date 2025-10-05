// Source-based slice around line 943
// Method: <com.google.common.reflect.TypeToken: boolean isSupertypeOfArray(GenericArrayType)>

    } else if (runtimeType instanceof GenericArrayType) {
      GenericArrayType fromArrayType = (GenericArrayType) runtimeType;
      return of(fromArrayType.getGenericComponentType())
          .isSubtypeOf(supertype.getGenericComponentType());
    } else {
      return false;
    }
  }

  private boolean isSupertypeOfArray(GenericArrayType subtype) {
    if (runtimeType instanceof Class) {
      Class<?> thisClass = (Class<?>) runtimeType;
      if (!thisClass.isArray()) {
        return thisClass.isAssignableFrom(Object[].class);
      }
      return of(subtype.getGenericComponentType()).isSubtypeOf(thisClass.getComponentType());
    } else if (runtimeType instanceof GenericArrayType) {
      return of(subtype.getGenericComponentType())
          .isSubtypeOf(((GenericArrayType) runtimeType).getGenericComponentType());
    } else {
