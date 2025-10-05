// Source-based slice around line 893
// Method: <com.google.common.reflect.TypeToken: boolean someRawTypeIsSubclassOf(Class)>


      @Override
      void visitGenericArrayType(GenericArrayType type) {
        visit(type.getGenericComponentType());
      }
    }.visit(runtimeType);
    return this;
  }

  private boolean someRawTypeIsSubclassOf(Class<?> superclass) {
    for (Class<?> rawType : getRawTypes()) {
      if (superclass.isAssignableFrom(rawType)) {
        return true;
      }
    }
    return false;
  }

  private boolean isSubtypeOfParameterizedType(ParameterizedType supertype) {
    Class<?> matchedClass = of(supertype).getRawType();
