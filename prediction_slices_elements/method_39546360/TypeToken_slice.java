// Source-based slice around line 1101
// Method: <com.google.common.reflect.TypeToken: ImmutableSet getRawTypes()>

      for (Type bound : bounds) {
        if (type.isSubtypeOf(bound) == target) {
          return target;
        }
      }
      return !target;
    }
  }

  private ImmutableSet<Class<? super T>> getRawTypes() {
    ImmutableSet.Builder<Class<?>> builder = ImmutableSet.builder();
    new TypeVisitor() {
      @Override
      void visitTypeVariable(TypeVariable<?> t) {
        visit(t.getBounds());
      }

      @Override
      void visitWildcardType(WildcardType t) {
        visit(t.getUpperBounds());
