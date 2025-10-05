// Source-based slice around line 105
// Method: <com.google.common.graph.Graphs: boolean subgraphHasCycle(Graph,Map,N)>

    return hasCycle(network.asGraph());
  }

  /**
   * Performs a traversal of the nodes reachable from {@code startNode}. If we ever reach a node
   * we've already visited (following only outgoing edges and without reusing edges), we know
   * there's a cycle in the graph.
   */
  private static <N> boolean subgraphHasCycle(
      Graph<N> graph, Map<Object, NodeVisitState> visitedNodes, N startNode) {
    Deque<NodeAndRemainingSuccessors<N>> stack = new ArrayDeque<>();
    stack.addLast(new NodeAndRemainingSuccessors<>(startNode));

    while (!stack.isEmpty()) {
      // To peek at the top two items, we need to temporarily remove one.
      NodeAndRemainingSuccessors<N> top = stack.removeLast();
      NodeAndRemainingSuccessors<N> prev = stack.peekLast();
      stack.addLast(top);

      N node = top.node;
