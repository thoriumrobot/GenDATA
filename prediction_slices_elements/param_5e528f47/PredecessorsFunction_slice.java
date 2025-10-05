// Source-based slice around line 104
// Method: <com.google.common.graph.PredecessorsFunction: Iterable predecessors(N)>

   * <ul>
   *   <li>Non-null
   *   <li>Usable as {@code Map} keys (see the Guava User Guide's section on <a
   *       href="https://github.com/google/guava/wiki/GraphsExplained#graph-elements-nodes-and-edges">
   *       graph elements</a> for details)
   * </ul>
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this graph
   */
  Iterable<? extends N> predecessors(N node);
}
