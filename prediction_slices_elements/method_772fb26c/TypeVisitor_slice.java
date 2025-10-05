// Source-based slice around line 65
// Method: <com.google.common.reflect.TypeVisitor: void visit(Type)>

 */
abstract class TypeVisitor {

  private final Set<Type> visited = new HashSet<>();

  /**
   * Visits the given types. Null types are ignored. This allows subclasses to call {@code
   * visit(parameterizedType.getOwnerType())} safely without having to check nulls.
   */
  public final void visit(@Nullable Type... types) {
    for (Type type : types) {
      if (type == null || !visited.add(type)) {
        // null owner type, or already visited;
        continue;
      }
      boolean succeeded = false;
      try {
        if (type instanceof TypeVariable) {
          visitTypeVariable((TypeVariable<?>) type);
        } else if (type instanceof WildcardType) {
