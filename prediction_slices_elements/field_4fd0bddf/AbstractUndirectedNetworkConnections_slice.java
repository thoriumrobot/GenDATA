// Source-based slice around line 37
// Method: com.google.common.graph.AbstractUndirectedNetworkConnections.incidentEdgeMap

/**
 * A base implementation of {@link NetworkConnections} for undirected networks.
 *
 * @author James Sexton
 * @param <N> Node parameter type
 * @param <E> Edge parameter type
 */
abstract class AbstractUndirectedNetworkConnections<N, E> implements NetworkConnections<N, E> {
  /** Keys are edges incident to the origin node, values are the node at the other end. */
  final Map<E, N> incidentEdgeMap;

  AbstractUndirectedNetworkConnections(Map<E, N> incidentEdgeMap) {
    this.incidentEdgeMap = checkNotNull(incidentEdgeMap);
  }

  @Override
  public Set<N> predecessors() {
    return adjacentNodes();
  }

