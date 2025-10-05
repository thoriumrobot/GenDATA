// Source-based slice around line 199
// Method: <com.google.common.graph.NetworkBuilder: NetworkBuilder cast()>

    return newBuilder;
  }

  /** Returns an empty {@link MutableNetwork} with the properties of this {@link NetworkBuilder}. */
  public <N1 extends N, E1 extends E> MutableNetwork<N1, E1> build() {
    return new StandardMutableNetwork<>(this);
  }

  @SuppressWarnings("unchecked")
  private <N1 extends N, E1 extends E> NetworkBuilder<N1, E1> cast() {
    return (NetworkBuilder<N1, E1>) this;
  }
}
