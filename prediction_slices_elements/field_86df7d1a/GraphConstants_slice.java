// Source-based slice around line 51
// Method: com.google.common.graph.GraphConstants.SELF_LOOPS_NOT_ALLOWED

  static final String REUSING_EDGE =
      "Edge %s already exists between the following nodes: %s, "
          + "so it cannot be reused to connect the following nodes: %s.";
  static final String MULTIPLE_EDGES_CONNECTING =
      "Cannot call edgeConnecting() when parallel edges exist between %s and %s. Consider calling "
          + "edgesConnecting() instead.";
  static final String PARALLEL_EDGES_NOT_ALLOWED =
      "Nodes %s and %s are already connected by a different edge. To construct a graph "
          + "that allows parallel edges, call allowsParallelEdges(true) on the Builder.";
  static final String SELF_LOOPS_NOT_ALLOWED =
      "Cannot add self-loop edge on node %s, as self-loops are not allowed. To construct a graph "
          + "that allows self-loops, call allowsSelfLoops(true) on the Builder.";
  static final String NOT_AVAILABLE_ON_UNDIRECTED =
      "Cannot call source()/target() on a EndpointPair from an undirected graph. Consider calling "
          + "adjacentNode(node) if you already have a node, or nodeU()/nodeV() if you don't.";
  static final String EDGE_ALREADY_EXISTS = "Edge %s already exists in the graph.";
  static final String ENDPOINTS_MISMATCH =
      "Mismatch: endpoints' ordering is not compatible with directionality of the graph";

  /** Singleton edge value for {@link Graph} implementations backed by {@link ValueGraph}s. */
