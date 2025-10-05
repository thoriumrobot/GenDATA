// Source-based slice around line 205
// Method: <com.google.common.collect.MapMakerInternalMap: MapMakerInternalMap create(MapMaker)>

      segmentSize <<= 1;
    }

    for (int i = 0; i < this.segments.length; ++i) {
      this.segments[i] = createSegment(segmentSize);
    }
  }

  /** Returns a fresh {@link MapMakerInternalMap} as specified by the given {@code builder}. */
  static <K, V> MapMakerInternalMap<K, V, ? extends InternalEntry<K, V, ?>, ?> create(
      MapMaker builder) {
    if (builder.getKeyStrength() == Strength.STRONG
        && builder.getValueStrength() == Strength.STRONG) {
      return new MapMakerInternalMap<>(builder, StrongKeyStrongValueEntry.Helper.instance());
    }
    if (builder.getKeyStrength() == Strength.STRONG
        && builder.getValueStrength() == Strength.WEAK) {
      return new MapMakerInternalMap<>(builder, StrongKeyWeakValueEntry.Helper.instance());
    }
    if (builder.getKeyStrength() == Strength.WEAK
