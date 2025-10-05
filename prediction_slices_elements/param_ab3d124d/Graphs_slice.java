// Source-based slice around line 168
// Method: <com.google.common.graph.Graphs: boolean canTraverseWithoutReusingEdge(Graph,Object,Object)>

  }

  /**
   * Determines whether an edge has already been used during traversal. In the directed case a cycle
   * is always detected before reusing an edge, so no special logic is required. In the undirected
   * case, we must take care not to "backtrack" over an edge (i.e. going from A to B and then going
   * from B to A).
   */
  private static boolean canTraverseWithoutReusingEdge(
      Graph<?> graph, Object nextNode, @Nullable Object previousNode) {
    if (graph.isDirected() || !Objects.equals(previousNode, nextNode)) {
      return true;
    }
    // This falls into the undirected A->B->A case. The Graph interface does not support parallel
    // edges, so this traversal would require reusing the undirected AB edge.
    return false;
  }

  /**
   * Returns the transitive closure of {@code graph}. The transitive closure of a graph is another
