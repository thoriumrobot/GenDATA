// Source-based slice around line 121
// Method: <com.google.common.collect.testing.MapTestSuiteBuilderTests: Test testsForHashMapNullValuesForbidden()>

                return map.put(key, value);
              }
            };
          }
        },
        "HashMap w/out null keys",
        ALLOWS_NULL_VALUES);
  }

  private static Test testsForHashMapNullValuesForbidden() {
    return wrappedHashMapTests(
        new WrappedHashMapGenerator() {
          @Override
          Map<String, String> wrap(HashMap<String, String> map) {
            if (map.containsValue(null)) {
              throw new NullPointerException();
            }

            return new AbstractMap<String, String>() {
              @Override
