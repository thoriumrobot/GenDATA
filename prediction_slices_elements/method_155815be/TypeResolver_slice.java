// Source-based slice around line 123
// Method: <com.google.common.reflect.TypeResolver: void populateTypeMappings(Map,Type,Type)>

    populateTypeMappings(mappings, checkNotNull(formal), checkNotNull(actual));
    return where(mappings);
  }

  /** Returns a new {@code TypeResolver} with {@code variable} mapping to {@code type}. */
  TypeResolver where(Map<TypeVariableKey, ? extends Type> mappings) {
    return new TypeResolver(typeTable.where(mappings));
  }

  private static void populateTypeMappings(
      Map<TypeVariableKey, Type> mappings, Type from, Type to) {
    if (from.equals(to)) {
      return;
    }
    new TypeVisitor() {
      @Override
      void visitTypeVariable(TypeVariable<?> typeVariable) {
        mappings.put(new TypeVariableKey(typeVariable), to);
      }

