// Source-based slice around line 119
// Method: <com.google.common.reflect.TypeResolver: TypeResolver where(Map)>

   *     corresponding mappings exist in the current {@code TypeResolver} instance.
   */
  public TypeResolver where(Type formal, Type actual) {
    Map<TypeVariableKey, Type> mappings = new HashMap<>();
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
