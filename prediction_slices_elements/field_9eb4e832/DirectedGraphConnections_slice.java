// Source-based slice around line 122
// Method: com.google.common.graph.DirectedGraphConnections.PRED


      @Override
      public int hashCode() {
        // Adding the class hashCode to avoid a clash with Pred instances.
        return Succ.class.hashCode() + node.hashCode();
      }
    }
  }

  private static final Object PRED = new Object();

  // Every value in this map must either be an instance of PredAndSucc with a successorValue of
  // type V, PRED (representing predecessor), or an instance of type V (representing successor).
  private final Map<N, Object> adjacentNodeValues;

  /**
   * All node connections in this graph, in edge insertion order.
   *
   * <p>Note: This field and {@link #adjacentNodeValues} cannot be combined into a single
   * LinkedHashMap because one target node may be mapped to both a predecessor and a successor. A
