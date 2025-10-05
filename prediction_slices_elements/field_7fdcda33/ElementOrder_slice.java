// Source-based slice around line 50
// Method: com.google.common.graph.ElementOrder.type

 *     GraphBuilder.directed().nodeOrder(ElementOrder.<Integer>natural()).build();
 * }
 *
 * @author Joshua O'Madadhain
 * @since 20.0
 */
@Beta
@Immutable
public final class ElementOrder<T> {
  private final Type type;

  @SuppressWarnings("Immutable") // Hopefully the comparator provided is immutable!
  private final @Nullable Comparator<T> comparator;

  /**
   * The type of ordering that this object specifies.
   *
   * <ul>
   *   <li>UNORDERED: no order is guaranteed.
   *   <li>STABLE: ordering is guaranteed to follow a pattern that won't change between releases.
