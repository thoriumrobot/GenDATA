// Source-based slice around line 102
// Method: <com.google.common.testing.TearDownStackTest: void runBare()>

      throw new RuntimeException(
          "A ClusterException should have been thrown, rather than a " + e.getClass().getName(), e);
    }

    assertEquals(true, tearDownOne.ran);
    assertEquals(true, tearDownTwo.ran);
  }

  @Override
  public final void runBare() throws Throwable {
    try {
      setUp();
      runTest();
    } finally {
      tearDown();
    }
  }

  @Override
  protected void tearDown() {
