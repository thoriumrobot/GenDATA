// Source-based slice around line 123
// Method: <com.google.common.collect.TreeRangeSet: Range rangeContaining(C)>

    }

    @Override
    public boolean equals(@Nullable Object o) {
      return Sets.equalsImpl(this, o);
    }
  }

  @Override
  public @Nullable Range<C> rangeContaining(C value) {
    checkNotNull(value);
    Entry<Cut<C>, Range<C>> floorEntry = rangesByLowerBound.floorEntry(Cut.belowValue(value));
    if (floorEntry != null && floorEntry.getValue().contains(value)) {
      return floorEntry.getValue();
    } else {
      // TODO(kevinb): revisit this design choice
      return null;
    }
  }

