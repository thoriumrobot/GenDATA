// Source-based slice around line 36
// Method: <com.google.common.graph.GraphConnections: Set predecessors()>

 *
 * @author James Sexton
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

