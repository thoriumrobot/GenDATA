// Source-based slice around line 35
// Method: <com.google.common.graph.NetworkConnections: Set predecessors()>

 *
 * @author James Sexton
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
