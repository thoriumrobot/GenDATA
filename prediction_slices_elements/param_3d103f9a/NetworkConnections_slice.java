// Source-based slice around line 49
// Method: <com.google.common.graph.NetworkConnections: Set edgesConnecting(N)>


  Set<E> inEdges();

  Set<E> outEdges();

  /**
   * Returns the set of edges connecting the origin node to {@code node}. For networks without
   * parallel edges, this set cannot be of size greater than one.
   */
  Set<E> edgesConnecting(N node);

  /**
   * Returns the node that is adjacent to the origin node along {@code edge}.
   *
   * <p>In the directed case, {@code edge} is assumed to be an outgoing edge.
   */
  N adjacentNode(E edge);

  /**
   * Remove {@code edge} from the set of incoming edges. Returns the former predecessor node.
