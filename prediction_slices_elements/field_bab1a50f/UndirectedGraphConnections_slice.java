// Source-based slice around line 41
// Method: com.google.common.graph.UndirectedGraphConnections.adjacentNodeValues


/**
 * An implementation of {@link GraphConnections} for undirected graphs.
 *
 * @author James Sexton
 * @param <N> Node parameter type
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
