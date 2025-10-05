// Source-based slice around line 320
// Method: <com.google.common.graph.DirectedGraphConnections: Set successors()>


      @Override
      public boolean contains(@Nullable Object obj) {
        return isPredecessor(adjacentNodeValues.get(obj));
      }
    };
  }

  @Override
  public Set<N> successors() {
    return new AbstractSet<N>() {
      @Override
      public UnmodifiableIterator<N> iterator() {
        if (orderedNodeConnections == null) {
          Iterator<Entry<N, Object>> entries = adjacentNodeValues.entrySet().iterator();
          return new AbstractIterator<N>() {
            @Override
            protected @Nullable N computeNext() {
              while (entries.hasNext()) {
                Entry<N, Object> entry = entries.next();
