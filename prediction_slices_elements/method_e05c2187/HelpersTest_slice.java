// Source-based slice around line 74
// Method: <com.google.common.collect.testing.HelpersTest: void testIsEmpty_map()>

            public Iterator<String> iterator() {
              return singleton("a").iterator();
            }
          });
      throw new Error();
    } catch (AssertionFailedError expected) {
    }
  }

  public void testIsEmpty_map() {
    Map<Object, Object> map = new HashMap<>();
    assertEmpty(map);

    map.put("a", "b");
    try {
      assertEmpty(map);
      throw new Error();
    } catch (AssertionFailedError expected) {
    }
  }
