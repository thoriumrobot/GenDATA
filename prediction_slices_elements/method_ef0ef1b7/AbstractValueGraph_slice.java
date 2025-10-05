// Source-based slice around line 137
// Method: <com.google.common.graph.AbstractValueGraph: int hashCode()>

    }
    ValueGraph<?, ?> other = (ValueGraph<?, ?>) obj;

    return isDirected() == other.isDirected()
        && nodes().equals(other.nodes())
        && edgeValueMap(this).equals(edgeValueMap(other));
  }

  @Override
  public final int hashCode() {
    return edgeValueMap(this).hashCode();
  }

  /** Returns a string representation of this graph. */
  @Override
  public String toString() {
    return "isDirected: "
        + isDirected()
        + ", allowsSelfLoops: "
        + allowsSelfLoops()
