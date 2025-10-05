// Source-based slice around line 107
// Method: <com.google.common.graph.SuccessorsFunction: Iterable successors(N)>

   * <ul>
   *   <li>Non-null
   *   <li>Usable as {@code Map} keys (see the Guava User Guide's section on <a
   *       href="https://github.com/google/guava/wiki/GraphsExplained#graph-elements-nodes-and-edges">
   *       graph elements</a> for details)
   * </ul>
   *
   * @throws IllegalArgumentException if {@code node} is not an element of this graph
   */
  Iterable<? extends N> successors(N node);
}
