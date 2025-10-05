// Source-based slice around line 166
// Method: <com.google.common.cache.CacheBuilderSpec: CacheBuilderSpec disableCaching()>

        String value = keyAndValue.size() == 1 ? null : keyAndValue.get(1);
        valueParser.parse(spec, key, value);
      }
    }

    return spec;
  }

  /** Returns a CacheBuilderSpec that will prevent caching. */
  public static CacheBuilderSpec disableCaching() {
    // Maximum size of zero is one way to block caching
    return CacheBuilderSpec.parse("maximumSize=0");
  }

  /** Returns a CacheBuilder configured according to this instance's specification. */
  CacheBuilder<Object, Object> toCacheBuilder() {
    CacheBuilder<Object, Object> builder = CacheBuilder.newBuilder();
    if (initialCapacity != null) {
      builder.initialCapacity(initialCapacity);
    }
