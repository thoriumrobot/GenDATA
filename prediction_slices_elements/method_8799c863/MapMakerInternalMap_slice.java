// Source-based slice around line 2346
// Method: <com.google.common.collect.MapMakerInternalMap: int size()>

        }
        sum -= segments[i].modCount;
      }
      return sum == 0L;
    }
    return true;
  }

  @Override
  public int size() {
    Segment<K, V, E, S>[] segments = this.segments;
    long sum = 0;
    for (int i = 0; i < segments.length; ++i) {
      sum += segments[i].count;
    }
    return Ints.saturatedCast(sum);
  }

  @Override
  public @Nullable V get(@Nullable Object key) {
