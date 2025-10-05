// Source-based slice around line 98
// Method: <com.google.common.graph.Traverser: Traverser forGraph(SuccessorsFunction)>

   *       href="https://github.com/google/guava/wiki/GraphsExplained#elements-must-be-useable-as-map-keys">
   *       notes on element objects</a> for more information.)
   *   <li>While traversing, the traverser will use <i>O(n)</i> space (where <i>n</i> is the number
   *       of nodes that have thus far been visited), plus <i>O(H)</i> space (where <i>H</i> is the
   *       number of nodes that have been seen but not yet visited, that is, the "horizon").
   * </ul>
   *
   * @param graph {@link SuccessorsFunction} representing a general graph that may have cycles.
   */
  public static <N> Traverser<N> forGraph(SuccessorsFunction<N> graph) {
    return new Traverser<N>(graph) {
      @Override
      Traversal<N> newTraversal() {
        return Traversal.inGraph(graph);
      }
    };
  }

  /**
   * Creates a new traverser for a directed acyclic graph that has at most one path from the start
