// Source-based slice around line 33
// Method: <com.google.common.graph.BaseGraph: Set nodes()>

 * @author James Sexton
 * @param <N> Node parameter type
 */
interface BaseGraph<N> extends SuccessorsFunction<N>, PredecessorsFunction<N> {
  //
  // Graph-level accessors
  //

  /** Returns all nodes in this graph, in the order specified by {@link #nodeOrder()}. */
  Set<N> nodes();

  /** Returns all edges in this graph. */
  Set<EndpointPair<N>> edges();

  //
  // Graph properties
  //

  /**
   * Returns true if the edges in this graph are directed. Directed edges connect a {@link
