// Source-based slice around line 927
// Method: <com.google.common.reflect.TypeToken: boolean isSubtypeOfArrayType(GenericArrayType)>

    }
    // We only care about the case when the supertype is a non-static inner class
    // in which case we need to make sure the subclass's owner type is a subtype of the
    // supertype's owner.
    return Modifier.isStatic(((Class<?>) supertype.getRawType()).getModifiers())
        || supertype.getOwnerType() == null
        || isOwnedBySubtypeOf(supertype.getOwnerType());
  }

  private boolean isSubtypeOfArrayType(GenericArrayType supertype) {
    if (runtimeType instanceof Class) {
      Class<?> fromClass = (Class<?>) runtimeType;
      if (!fromClass.isArray()) {
        return false;
      }
      return of(fromClass.getComponentType()).isSubtypeOf(supertype.getGenericComponentType());
    } else if (runtimeType instanceof GenericArrayType) {
      GenericArrayType fromArrayType = (GenericArrayType) runtimeType;
      return of(fromArrayType.getGenericComponentType())
          .isSubtypeOf(supertype.getGenericComponentType());
