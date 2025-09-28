package org.checkerframework.specimin;

import java.io.IOException;
import org.junit.Test;

/**
 * This test checks that Specimin can correctly create an interface to be the least upper bound of
 * multiple RHSs that are assigned to the same field of unknown type in a synthetic superclass.
 */
public class SyntheticSuperLub {
  @Test
  public void runTest() throws IOException {
    SpeciminTestExecutor.runTestWithoutJarPaths(
        "syntheticsuperlub",
        new String[] {"com/example/Dog.java"},
        new String[] {"com.example.Dog#setup(String)"});
  }
}
