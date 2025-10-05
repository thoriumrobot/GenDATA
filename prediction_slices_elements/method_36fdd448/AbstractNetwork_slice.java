// Source-based slice around line 275
// Method: <com.google.common.graph.AbstractNetwork: String toString()>

  }

  @Override
  public final int hashCode() {
    return edgeIncidentNodesMap(this).hashCode();
  }

  /** Returns a string representation of this network. */
  @Override
  public String toString() {
    return "isDirected: "
        + isDirected()
        + ", allowsParallelEdges: "
        + allowsParallelEdges()
        + ", allowsSelfLoops: "
        + allowsSelfLoops()
        + ", nodes: "
        + nodes()
        + ", edges: "
        + edgeIncidentNodesMap(this);
