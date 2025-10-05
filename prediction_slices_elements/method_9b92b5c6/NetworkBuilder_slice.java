// Source-based slice around line 119
// Method: <com.google.common.graph.NetworkBuilder: ImmutableNetwork immutable()>

  }

  /**
   * Returns an {@link ImmutableNetwork.Builder} with the properties of this {@link NetworkBuilder}.
   *
   * <p>The returned builder can be used for populating an {@link ImmutableNetwork}.
   *
   * @since 28.0
   */
  public <N1 extends N, E1 extends E> ImmutableNetwork.Builder<N1, E1> immutable() {
    NetworkBuilder<N1, E1> castBuilder = cast();
    return new ImmutableNetwork.Builder<>(castBuilder);
  }

  /**
   * Specifies whether the network will allow parallel edges. Attempting to add a parallel edge to a
   * network that does not allow them will throw an {@link UnsupportedOperationException}.
   *
   * <p>The default value is {@code false}.
   */
