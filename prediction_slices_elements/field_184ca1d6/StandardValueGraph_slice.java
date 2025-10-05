// Source-based slice around line 47
// Method: com.google.common.graph.StandardValueGraph.isDirected

 * <p>The time complexity of all collection-returning accessors is O(1), since views are returned.
 *
 * @author James Sexton
 * @author Joshua O'Madadhain
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
