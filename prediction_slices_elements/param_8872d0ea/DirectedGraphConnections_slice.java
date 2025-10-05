// Source-based slice around line 546
// Method: <com.google.common.graph.DirectedGraphConnections: boolean isPredecessor(Object)>

      if (orderedNodeConnections != null) {
        orderedNodeConnections.add(new NodeConnection.Succ<>(node));
      }
    }

    // See the comment on the similar cast in removeSuccessor.
    return previousSuccessor == null ? null : (V) previousSuccessor;
  }

  private static boolean isPredecessor(@Nullable Object value) {
    return (value == PRED) || (value instanceof PredAndSucc);
  }

  private static boolean isSuccessor(@Nullable Object value) {
    return (value != PRED) && (value != null);
  }
}
