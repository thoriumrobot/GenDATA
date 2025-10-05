// Source-based slice around line 269
// Method: <com.google.common.graph.AbstractNetwork: int hashCode()>

    }
    Network<?, ?> other = (Network<?, ?>) obj;

    return isDirected() == other.isDirected()
        && nodes().equals(other.nodes())
        && edgeIncidentNodesMap(this).equals(edgeIncidentNodesMap(other));
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
