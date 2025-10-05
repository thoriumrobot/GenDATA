// Source-based slice around line 1039
// Method: <com.google.common.reflect.TypeToken: WildcardType canonicalizeWildcardType(TypeVariable,WildcardType)>

      return Types.newArrayType(
          canonicalizeWildcardsInType(((GenericArrayType) type).getGenericComponentType()));
    }
    return type;
  }

  // WARNING: the returned type may have empty upper bounds, which may violate common expectations
  // by user code or even some of our own code. It's fine for the purpose of checking subtypes.
  // Just don't ever let the user access it.
  private static WildcardType canonicalizeWildcardType(
      TypeVariable<?> declaration, WildcardType type) {
    Type[] declared = declaration.getBounds();
    List<Type> upperBounds = new ArrayList<>();
    for (Type bound : type.getUpperBounds()) {
      if (!any(declared).isSubtypeOf(bound)) {
        upperBounds.add(canonicalizeWildcardsInType(bound));
      }
    }
    return new Types.WildcardTypeImpl(type.getLowerBounds(), upperBounds.toArray(new Type[0]));
  }
