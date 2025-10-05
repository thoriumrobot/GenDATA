// Source-based slice around line 53
// Method: com.google.common.graph.ElementOrder.comparator

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
   *       Some methods may have stronger guarantees.
   *   <li>INSERTION: insertion ordering is guaranteed.
   *   <li>SORTED: ordering according to a supplied comparator is guaranteed.
