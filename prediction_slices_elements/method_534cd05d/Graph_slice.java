// Source-based slice around line 113
// Method: <com.google.common.graph.Graph: Set nodes()>

@Beta
@DoNotMock("Use GraphBuilder to create a real instance")
public interface Graph<N> extends BaseGraph<N> {
  //
  // Graph-level accessors
  //

  /** Returns all nodes in this graph, in the order specified by {@link #nodeOrder()}. */
  @Override
  Set<N> nodes();

  /** Returns all edges in this graph. */
  @Override
  Set<EndpointPair<N>> edges();

  //
  // Graph properties
  //

  /**
