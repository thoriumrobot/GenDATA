// Source-based slice around line 902
// Method: <com.google.common.reflect.TypeToken: boolean isSubtypeOfParameterizedType(ParameterizedType)>

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
    if (!someRawTypeIsSubclassOf(matchedClass)) {
      return false;
    }
    TypeVariable<?>[] typeVars = matchedClass.getTypeParameters();
    Type[] supertypeArgs = supertype.getActualTypeArguments();
    for (int i = 0; i < typeVars.length; i++) {
      Type subtypeParam = getCovariantTypeResolver().resolveType(typeVars[i]);
      // If 'supertype' is "List<? extends CharSequence>"
      // and 'this' is StringArrayList,
