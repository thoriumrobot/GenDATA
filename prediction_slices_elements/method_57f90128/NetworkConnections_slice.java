// Source-based slice around line 41
// Method: <com.google.common.graph.NetworkConnections: Set inEdges()>


  Set<N> adjacentNodes();

  Set<N> predecessors();

  Set<N> successors();

  Set<E> incidentEdges();

  Set<E> inEdges();

  Set<E> outEdges();

  /**
   * Returns the set of edges connecting the origin node to {@code node}. For networks without
   * parallel edges, this set cannot be of size greater than one.
   */
  Set<E> edgesConnecting(N node);

  /**
