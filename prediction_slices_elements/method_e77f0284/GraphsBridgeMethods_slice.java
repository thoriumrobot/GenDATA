// Source-based slice around line 19
// Method: <com.google.common.graph.GraphsBridgeMethods: Set reachableNodes(Graph,N)>

@Beta
abstract class GraphsBridgeMethods {

  @SuppressWarnings("PreferredInterfaceType")
  public static <N> Graph<N> transitiveClosure(Graph<N> graph) {
    return Graphs.transitiveClosure(graph);
  }

  @SuppressWarnings("PreferredInterfaceType")
  public static <N> Set<N> reachableNodes(Graph<N> graph, N node) {
    return Graphs.reachableNodes(graph, node);
  }
}
