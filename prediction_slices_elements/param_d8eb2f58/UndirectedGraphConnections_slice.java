// Source-based slice around line 60
// Method: <com.google.common.graph.UndirectedGraphConnections: UndirectedGraphConnections ofImmutable(Map)>

            new HashMap<N, V>(INNER_CAPACITY, INNER_LOAD_FACTOR));
      case STABLE:
        return new UndirectedGraphConnections<>(
            new LinkedHashMap<N, V>(INNER_CAPACITY, INNER_LOAD_FACTOR));
      default:
        throw new AssertionError(incidentEdgeOrder.type());
    }
  }

  static <N, V> UndirectedGraphConnections<N, V> ofImmutable(Map<N, V> adjacentNodeValues) {
    return new UndirectedGraphConnections<>(ImmutableMap.copyOf(adjacentNodeValues));
  }

  @Override
  public Set<N> adjacentNodes() {
    return Collections.unmodifiableSet(adjacentNodeValues.keySet());
  }

  @Override
  public Set<N> predecessors() {
