// Source-based slice around line 115
// Method: <com.google.common.graph.Network: Set nodes()>

 */
@Beta
@DoNotMock("Use NetworkBuilder to create a real instance")
public interface Network<N, E> extends SuccessorsFunction<N>, PredecessorsFunction<N> {
  //
  // Network-level accessors
  //

  /** Returns all nodes in this network, in the order specified by {@link #nodeOrder()}. */
  Set<N> nodes();

  /** Returns all edges in this network, in the order specified by {@link #edgeOrder()}. */
  Set<E> edges();

  /**
   * Returns a live view of this network as a {@link Graph}. The resulting {@link Graph} will have
   * an edge connecting node A to node B if this {@link Network} has an edge connecting A to B.
   *
   * <p>If this network {@link #allowsParallelEdges() allows parallel edges}, parallel edges will be
   * treated as if collapsed into a single edge. For example, the {@link #degree(Object)} of a node
