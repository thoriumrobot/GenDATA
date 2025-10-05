// Source-based slice around line 343
// Method: <com.google.common.collect.testing.MapTestSuiteBuilderTests: LinkageError newLinkageError(Throwable)>

            MapFeature.ALLOWS_NULL_VALUES,
            CollectionFeature.SERIALIZABLE,
            CollectionFeature.SUPPORTS_ITERATOR_REMOVE,
            CollectionSize.ANY)
        .withSetUp(setUp)
        .withTearDown(tearDown)
        .createTestSuite();
  }

  private static LinkageError newLinkageError(Throwable cause) {
    return new LinkageError(cause.toString(), cause);
  }
}
