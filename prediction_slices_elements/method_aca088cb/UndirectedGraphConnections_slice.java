// Source-based slice around line 47
// Method: <com.google.common.graph.UndirectedGraphConnections: UndirectedGraphConnections of(ElementOrder)>

 * @param <V> Value parameter type
 */
final class UndirectedGraphConnections<N, V> implements GraphConnections<N, V> {
  private final Map<N, V> adjacentNodeValues;

  private UndirectedGraphConnections(Map<N, V> adjacentNodeValues) {
    this.adjacentNodeValues = checkNotNull(adjacentNodeValues);
  }

  static <N, V> UndirectedGraphConnections<N, V> of(ElementOrder<N> incidentEdgeOrder) {
    switch (incidentEdgeOrder.type()) {
      case UNORDERED:
        return new UndirectedGraphConnections<>(
            new HashMap<N, V>(INNER_CAPACITY, INNER_LOAD_FACTOR));
      case STABLE:
        return new UndirectedGraphConnections<>(
            new LinkedHashMap<N, V>(INNER_CAPACITY, INNER_LOAD_FACTOR));
      default:
        throw new AssertionError(incidentEdgeOrder.type());
    }
