// Source-based slice around line 52
// Method: com.google.common.graph.StandardNetwork.isDirected

 * <p>The time complexity of all collection-returning accessors is O(1), since views are returned.
 *
 * @author James Sexton
 * @author Joshua O'Madadhain
 * @author Omar Darwish
 * @param <N> Node parameter type
 * @param <E> Edge parameter type
 */
class StandardNetwork<N, E> extends AbstractNetwork<N, E> {
  private final boolean isDirected;
  private final boolean allowsParallelEdges;
  private final boolean allowsSelfLoops;
  private final ElementOrder<N> nodeOrder;
  private final ElementOrder<E> edgeOrder;

  final MapIteratorCache<N, NetworkConnections<N, E>> nodeConnections;

  // We could make this a Map<E, EndpointPair<N>>. It would make incidentNodes(edge) slightly
  // faster, but also make Networks consume 5 to 20+% (increasing with average degree) more memory.
  final MapIteratorCache<E, N> edgeToReferenceNode; // referenceNode == source if directed
