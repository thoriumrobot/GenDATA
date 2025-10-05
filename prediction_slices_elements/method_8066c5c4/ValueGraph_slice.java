// Source-based slice around line 118
// Method: <com.google.common.graph.ValueGraph: Set nodes()>

 */
@Beta
public interface ValueGraph<N, V> extends BaseGraph<N> {
  //
  // ValueGraph-level accessors
  //

  /** Returns all nodes in this graph, in the order specified by {@link #nodeOrder()}. */
  @Override
  Set<N> nodes();

  /** Returns all edges in this graph. */
  @Override
  Set<EndpointPair<N>> edges();

  /**
   * Returns a live view of this graph as a {@link Graph}. The resulting {@link Graph} will have an
   * edge connecting node A to node B if this {@link ValueGraph} has an edge connecting A to B.
   */
  Graph<N> asGraph();
