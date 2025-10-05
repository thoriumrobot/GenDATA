// Source-based slice around line 153
// Method: <com.google.common.collect.ContiguousSet: ContiguousSet headSet(C)>


  final DiscreteDomain<C> domain;

  ContiguousSet(DiscreteDomain<C> domain) {
    super(Ordering.natural());
    this.domain = domain;
  }

  @Override
  public ContiguousSet<C> headSet(C toElement) {
    return headSetImpl(checkNotNull(toElement), false);
  }

  /**
   * @since 12.0
   */
  @GwtIncompatible // NavigableSet
  @Override
  public ContiguousSet<C> headSet(C toElement, boolean inclusive) {
    return headSetImpl(checkNotNull(toElement), inclusive);
