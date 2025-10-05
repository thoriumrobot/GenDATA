// Source-based slice around line 34
// Method: <com.google.common.graph.GraphConnections: Set adjacentNodes()>

 * An interface for representing and manipulating an origin node's adjacent nodes and edge values in
 * a {@link Graph}.
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
