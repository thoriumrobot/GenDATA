// Source-based slice around line 138
// Method: com.google.common.graph.DirectedGraphConnections.predecessorCount

   * All node connections in this graph, in edge insertion order.
   *
   * <p>Note: This field and {@link #adjacentNodeValues} cannot be combined into a single
   * LinkedHashMap because one target node may be mapped to both a predecessor and a successor. A
   * LinkedHashMap combines two such edges into a single node-value pair, even though the edges may
   * not have been inserted consecutively.
   */
  private final @Nullable List<NodeConnection<N>> orderedNodeConnections;

  private int predecessorCount;
  private int successorCount;

  private DirectedGraphConnections(
      Map<N, Object> adjacentNodeValues,
      @Nullable List<NodeConnection<N>> orderedNodeConnections,
      int predecessorCount,
      int successorCount) {
    this.adjacentNodeValues = checkNotNull(adjacentNodeValues);
    this.orderedNodeConnections = orderedNodeConnections;
    this.predecessorCount = checkNonNegative(predecessorCount);
