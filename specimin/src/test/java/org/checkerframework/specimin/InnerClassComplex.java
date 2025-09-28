package org.checkerframework.specimin;

import java.io.IOException;
import org.junit.Test;

/**
 * This test checks that if Specimin will work properly where there are inner classes in the input
 * code. This variant is more complex, based on an inner class in
 * org.checkerframework.dataflow.analysis.AbstractAnalysis.
 */
public class InnerClassComplex {
  @Test
  public void runTest() throws IOException {
    SpeciminTestExecutor.runTestWithoutJarPaths(
        "innerclasscomplex",
        new String[] {"com/example/AbstractAnalysis.java"},
        new String[] {
          "com.example.AbstractAnalysis.Worklist.ForwardDFOComparator#compare(Block, Block)"
        });
  }
}
