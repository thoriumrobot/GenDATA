// Source-based slice around line 38
// Method: <com.google.common.graph.GraphConnections: Set successors()>

 * @param <N> Node parameter type
 * @param <V> Value parameter type
 */
interface GraphConnections<N, V> {

  Set<N> adjacentNodes();

  Set<N> predecessors();

  Set<N> successors();

  /**
   * Returns an iterator over the incident edges.
   *
   * @param thisNode The node that this all of the connections in this class are connected to.
   */
  Iterator<EndpointPair<N>> incidentEdgeIterator(N thisNode);

  /**
   * Returns the value associated with the edge connecting the origin node to {@code node}, or null
