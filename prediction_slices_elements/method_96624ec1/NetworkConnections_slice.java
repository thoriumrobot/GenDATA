// Source-based slice around line 37
// Method: <com.google.common.graph.NetworkConnections: Set successors()>

 * @param <N> Node parameter type
 * @param <E> Edge parameter type
 */
interface NetworkConnections<N, E> {

  Set<N> adjacentNodes();

  Set<N> predecessors();

  Set<N> successors();

  Set<E> incidentEdges();

  Set<E> inEdges();

  Set<E> outEdges();

  /**
   * Returns the set of edges connecting the origin node to {@code node}. For networks without
   * parallel edges, this set cannot be of size greater than one.
