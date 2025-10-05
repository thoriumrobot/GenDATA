// Source-based slice around line 42
// Method: com.google.common.graph.GraphConstants.REUSING_EDGE

  // Error messages
  static final String NODE_NOT_IN_GRAPH = "Node %s is not an element of this graph.";
  static final String EDGE_NOT_IN_GRAPH = "Edge %s is not an element of this graph.";
  static final String NODE_REMOVED_FROM_GRAPH =
      "Node %s that was used to generate this set is no longer in the graph.";
  static final String NODE_PAIR_REMOVED_FROM_GRAPH =
      "Node %s or node %s that were used to generate this set are no longer in the graph.";
  static final String EDGE_REMOVED_FROM_GRAPH =
      "Edge %s that was used to generate this set is no longer in the graph.";
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
