// Source-based slice around line 41
// Method: <com.google.common.graph.DirectedNetworkConnections: DirectedNetworkConnections of()>

 * @param <N> Node parameter type
 * @param <E> Edge parameter type
 */
final class DirectedNetworkConnections<N, E> extends AbstractDirectedNetworkConnections<N, E> {

  DirectedNetworkConnections(Map<E, N> inEdgeMap, Map<E, N> outEdgeMap, int selfLoopCount) {
    super(inEdgeMap, outEdgeMap, selfLoopCount);
  }

  static <N, E> DirectedNetworkConnections<N, E> of() {
    return new DirectedNetworkConnections<>(
        HashBiMap.<E, N>create(EXPECTED_DEGREE), HashBiMap.<E, N>create(EXPECTED_DEGREE), 0);
  }

  static <N, E> DirectedNetworkConnections<N, E> ofImmutable(
      Map<E, N> inEdges, Map<E, N> outEdges, int selfLoopCount) {
    return new DirectedNetworkConnections<>(
        ImmutableBiMap.copyOf(inEdges), ImmutableBiMap.copyOf(outEdges), selfLoopCount);
  }

