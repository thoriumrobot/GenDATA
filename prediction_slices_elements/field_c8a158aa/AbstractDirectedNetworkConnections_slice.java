// Source-based slice around line 50
// Method: com.google.common.graph.AbstractDirectedNetworkConnections.selfLoopCount

 * @param <E> Edge parameter type
 */
abstract class AbstractDirectedNetworkConnections<N, E> implements NetworkConnections<N, E> {
  /** Keys are edges incoming to the origin node, values are the source node. */
  final Map<E, N> inEdgeMap;

  /** Keys are edges outgoing from the origin node, values are the target node. */
  final Map<E, N> outEdgeMap;

  private int selfLoopCount;

  AbstractDirectedNetworkConnections(Map<E, N> inEdgeMap, Map<E, N> outEdgeMap, int selfLoopCount) {
    this.inEdgeMap = checkNotNull(inEdgeMap);
    this.outEdgeMap = checkNotNull(outEdgeMap);
    this.selfLoopCount = checkNonNegative(selfLoopCount);
    checkState(selfLoopCount <= inEdgeMap.size() && selfLoopCount <= outEdgeMap.size());
  }

  @Override
  public Set<N> adjacentNodes() {
