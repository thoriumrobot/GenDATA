// Source-based slice around line 44
// Method: <com.google.common.graph.AbstractUndirectedNetworkConnections: Set predecessors()>

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

  @Override
  public Set<N> successors() {
    return adjacentNodes();
  }

  @Override
  public Set<E> incidentEdges() {
