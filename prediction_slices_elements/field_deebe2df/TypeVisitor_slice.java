// Source-based slice around line 59
// Method: com.google.common.reflect.TypeVisitor.visited

 * <p>One {@code Type} is visited at most once. The second time the same type is visited, it's
 * ignored by {@link #visit}. This avoids infinite recursion caused by recursive type bounds.
 *
 * <p>This class is <em>not</em> thread safe.
 *
 * @author Ben Yu
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
