// Source-based slice around line 51
// Method: com.google.common.graph.StandardValueGraph.nodeConnections

 * @author Omar Darwish
 * @param <N> Node parameter type
 * @param <V> Value parameter type
 */
class StandardValueGraph<N, V> extends AbstractValueGraph<N, V> {
  private final boolean isDirected;
  private final boolean allowsSelfLoops;
  private final ElementOrder<N> nodeOrder;

  final MapIteratorCache<N, GraphConnections<N, V>> nodeConnections;

  long edgeCount; // must be updated when edges are added or removed

  /** Constructs a graph with the properties specified in {@code builder}. */
  StandardValueGraph(AbstractGraphBuilder<? super N> builder) {
    this(
        builder,
        builder.nodeOrder.<N, GraphConnections<N, V>>createMap(
            builder.expectedNodeCount.or(DEFAULT_NODE_COUNT)),
        0L);
