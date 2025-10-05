/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xalan.internal.xsltc.compiler;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;

    @Positive
final class QName {

    @Positive
    public QName(String namespace, String prefix, String localname) {
    @Positive
    }

    @Positive
    public void clearNamespace();

    @Positive
    public String toString();

    @Positive
    public String getStringRep();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object other);

    @Positive
    public String getLocalPart();

    @Positive
    public String getNamespace();

    @Positive
    public String getPrefix();

    @Positive
    public int hashCode();

    @Positive
    public String dump();
    @Positive
}

// CFWR semantic augmentation - variant 0
