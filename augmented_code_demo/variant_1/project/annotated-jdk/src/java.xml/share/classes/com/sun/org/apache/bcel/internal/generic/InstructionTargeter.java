/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.bcel.internal.generic;

    @Positive
import org.checkerframework.dataflow.qual.Pure;

    @Positive
public interface InstructionTargeter {

    @Positive
    @Pure
    @Positive
    boolean containsTarget(InstructionHandle ih);

    @Positive
    void updateTarget(InstructionHandle old_ih, InstructionHandle new_ih) throws ClassGenException;
    @Positive
}

// CFWR semantic augmentation - variant 1
